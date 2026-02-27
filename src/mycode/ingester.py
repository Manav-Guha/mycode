"""Project Ingester (C2) — Static analysis and dependency resolution for Python projects.

AST parsing, dependency extraction, function flow mapping, and coupling point
identification. Handles partial parsing gracefully — reports what couldn't be
parsed and proceeds with what it can.

Pure Python. No LLM dependency.
"""

import ast
import configparser
import importlib.metadata
import json
import logging
import re
import urllib.error
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── Encoding-safe file reading ──

# BOM signatures for UTF-16 variants
_UTF16_LE_BOM = b"\xff\xfe"
_UTF16_BE_BOM = b"\xfe\xff"


def _read_text_safe(path: Path) -> str:
    """Read a text file, handling UTF-8, UTF-16 (BOM), and latin-1 gracefully.

    Raises OSError if the file cannot be read at all.
    """
    raw = path.read_bytes()
    # Detect UTF-16 BOM before trying UTF-8
    if raw[:2] in (_UTF16_LE_BOM, _UTF16_BE_BOM):
        return raw.decode("utf-16")
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("latin-1")

# Try tomllib (Python 3.11+) then tomli for pyproject.toml parsing
try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib  # type: ignore[no-redefine]
    except ImportError:
        tomllib = None  # type: ignore[assignment]

# ── Constants ──

# Directories to skip when discovering Python files
DISCOVER_EXCLUDE_DIRS = {
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
    ".eggs",
    "build",
    "dist",
    ".DS_Store",
    ".claude",
}

# Directory name prefixes to exclude (matched with str.startswith)
DISCOVER_EXCLUDE_PREFIXES = ("pytest-of-",)

# Regex for extracting a package name from a PEP 508 dependency string
_PKG_NAME_RE = re.compile(r"^([A-Za-z0-9]([A-Za-z0-9._-]*[A-Za-z0-9])?)")


# ── Exceptions ──


class IngestionError(Exception):
    """Base exception for ingestion errors."""


# ── Data Classes ──


@dataclass
class ImportInfo:
    """A single import statement extracted from a Python file."""

    module: str  # e.g. "os.path" or "flask"
    names: list[str] = field(default_factory=list)  # from-import names
    alias: Optional[str] = None  # e.g. "np" for import numpy as np
    is_from_import: bool = False
    lineno: int = 0


@dataclass
class FunctionInfo:
    """A function or method definition extracted from AST."""

    name: str
    file_path: str  # relative to project root
    lineno: int
    end_lineno: int = 0
    args: list[str] = field(default_factory=list)
    decorators: list[str] = field(default_factory=list)
    calls: list[str] = field(default_factory=list)  # names called within this function
    is_method: bool = False
    class_name: Optional[str] = None
    is_async: bool = False
    globals_accessed: list[str] = field(default_factory=list)  # from `global x` stmts


@dataclass
class ClassInfo:
    """A class definition extracted from AST."""

    name: str
    file_path: str
    lineno: int
    end_lineno: int = 0
    methods: list[str] = field(default_factory=list)
    bases: list[str] = field(default_factory=list)
    decorators: list[str] = field(default_factory=list)


@dataclass
class GlobalVarInfo:
    """A module-level variable assignment."""

    name: str
    file_path: str
    lineno: int


@dataclass
class FileAnalysis:
    """Analysis result for a single Python file."""

    file_path: str  # relative to project root
    functions: list[FunctionInfo] = field(default_factory=list)
    classes: list[ClassInfo] = field(default_factory=list)
    imports: list[ImportInfo] = field(default_factory=list)
    global_vars: list[GlobalVarInfo] = field(default_factory=list)
    parse_error: Optional[str] = None
    lines_of_code: int = 0


@dataclass
class DependencyInfo:
    """Information about a single project dependency."""

    name: str
    installed_version: Optional[str] = None
    latest_version: Optional[str] = None
    required_version: Optional[str] = None  # version spec from dep file
    source_file: Optional[str] = None  # which dep file declared this
    is_outdated: Optional[bool] = None  # True if installed < latest
    is_missing: bool = False  # True if declared but not installed
    is_dev: bool = False  # True if from devDependencies (JS only)


@dataclass
class FunctionFlow:
    """An edge in the function call graph."""

    caller: str  # "module.function" or "module.Class.method"
    callee: str  # resolved callee name
    file_path: str  # file where the call occurs
    lineno: int


@dataclass
class CouplingPoint:
    """A point where one component's failure could cascade into another."""

    source: str  # the originating function/module
    targets: list[str] = field(default_factory=list)  # affected functions/modules
    coupling_type: str = ""  # high_fan_in, cross_module_hub, shared_state
    description: str = ""  # human-readable explanation
    file_path: str = ""
    lineno: int = 0


@dataclass
class IngestionResult:
    """Complete result of project ingestion."""

    project_path: str
    files_analyzed: int = 0
    files_failed: int = 0
    total_lines: int = 0
    file_analyses: list[FileAnalysis] = field(default_factory=list)
    dependencies: list[DependencyInfo] = field(default_factory=list)
    dependency_tree: dict = field(default_factory=dict)  # {name: [transitive deps]}
    function_flows: list[FunctionFlow] = field(default_factory=list)
    coupling_points: list[CouplingPoint] = field(default_factory=list)
    parse_errors: list[dict] = field(default_factory=list)  # [{file, error}]
    warnings: list[str] = field(default_factory=list)


# ── Helpers ──


def _normalize_package_name(name: str) -> str:
    """Normalize a Python package name per PEP 503."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _parse_dep_string(dep_str: str) -> tuple[str, str]:
    """Parse a PEP 508 dependency string into (name, version_spec).

    Examples:
        "flask>=2.0"        -> ("flask", ">=2.0")
        "requests[security]" -> ("requests", "")
        "numpy==1.24; python_version >= '3.8'" -> ("numpy", "==1.24")
    """
    s = dep_str.strip()
    # Strip environment markers (after ;)
    if ";" in s:
        s = s[: s.index(";")].strip()
    match = re.match(r"^([A-Za-z0-9][A-Za-z0-9._-]*)(?:\[.*?\])?\s*(.*)", s)
    if match:
        return match.group(1), match.group(2).strip()
    return s, ""


def _is_version_outdated(installed: str, latest: str) -> bool:
    """Check if installed version is older than latest. Best-effort comparison."""
    try:
        inst_parts = [int(x) for x in installed.split(".")[:3]]
        latest_parts = [int(x) for x in latest.split(".")[:3]]
        while len(inst_parts) < 3:
            inst_parts.append(0)
        while len(latest_parts) < 3:
            latest_parts.append(0)
        return tuple(inst_parts) < tuple(latest_parts)
    except (ValueError, TypeError):
        return installed != latest


# ── AST Analyzer ──


class _FileAnalyzer(ast.NodeVisitor):
    """Walks a Python AST to extract functions, classes, imports, and call relationships."""

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.functions: list[FunctionInfo] = []
        self.classes: list[ClassInfo] = []
        self.imports: list[ImportInfo] = []
        self.global_vars: list[GlobalVarInfo] = []
        self._scope_stack: list[str] = []  # tracks current class/function scope
        self._current_function: Optional[FunctionInfo] = None

    def analyze(self, tree: ast.AST) -> FileAnalysis:
        """Run the analysis and return a FileAnalysis."""
        self.visit(tree)
        return FileAnalysis(
            file_path=self.file_path,
            functions=self.functions,
            classes=self.classes,
            imports=self.imports,
            global_vars=self.global_vars,
        )

    # ── Function/Method Definitions ──

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self._handle_function(node, is_async=False)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        self._handle_function(node, is_async=True)

    def _handle_function(self, node, is_async: bool):
        class_name = None
        is_method = False
        if self._scope_stack and self._scope_stack[-1].startswith("class:"):
            class_name = self._scope_stack[-1][6:]
            is_method = True

        func = FunctionInfo(
            name=node.name,
            file_path=self.file_path,
            lineno=node.lineno,
            end_lineno=getattr(node, "end_lineno", node.lineno),
            args=[arg.arg for arg in node.args.args],
            decorators=[self._get_decorator_name(d) for d in node.decorator_list],
            is_method=is_method,
            class_name=class_name,
            is_async=is_async,
        )

        # Collect calls and global accesses within this function body
        old_function = self._current_function
        self._current_function = func

        self._scope_stack.append(f"func:{node.name}")
        self.generic_visit(node)
        self._scope_stack.pop()

        self._current_function = old_function
        self.functions.append(func)

    # ── Class Definitions ──

    def visit_ClassDef(self, node: ast.ClassDef):
        cls = ClassInfo(
            name=node.name,
            file_path=self.file_path,
            lineno=node.lineno,
            end_lineno=getattr(node, "end_lineno", node.lineno),
            bases=[self._get_name(b) for b in node.bases],
            decorators=[self._get_decorator_name(d) for d in node.decorator_list],
        )

        func_count_before = len(self.functions)

        self._scope_stack.append(f"class:{node.name}")
        self.generic_visit(node)
        self._scope_stack.pop()

        # Collect method names added during this class visit
        cls.methods = [
            f.name
            for f in self.functions[func_count_before:]
            if f.class_name == node.name
        ]
        self.classes.append(cls)

    # ── Imports ──

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            self.imports.append(
                ImportInfo(
                    module=alias.name,
                    alias=alias.asname,
                    is_from_import=False,
                    lineno=node.lineno,
                )
            )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        module = node.module or ""
        names = [alias.name for alias in (node.names or [])]
        self.imports.append(
            ImportInfo(
                module=module,
                names=names,
                is_from_import=True,
                lineno=node.lineno,
            )
        )
        self.generic_visit(node)

    # ── Calls ──

    def visit_Call(self, node: ast.Call):
        if self._current_function is not None:
            name = self._get_call_name(node)
            if name:
                self._current_function.calls.append(name)
        self.generic_visit(node)

    # ── Global Keyword ──

    def visit_Global(self, node: ast.Global):
        if self._current_function is not None:
            self._current_function.globals_accessed.extend(node.names)
        self.generic_visit(node)

    # ── Module-Level Assignments ──

    def visit_Assign(self, node: ast.Assign):
        if not self._scope_stack:  # module level only
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.global_vars.append(
                        GlobalVarInfo(
                            name=target.id,
                            file_path=self.file_path,
                            lineno=node.lineno,
                        )
                    )
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign):
        if not self._scope_stack and node.target and isinstance(node.target, ast.Name):
            self.global_vars.append(
                GlobalVarInfo(
                    name=node.target.id,
                    file_path=self.file_path,
                    lineno=node.lineno,
                )
            )
        self.generic_visit(node)

    # ── Name Resolution Helpers ──

    def _get_call_name(self, node: ast.Call) -> str:
        return self._get_name(node.func)

    def _get_name(self, node) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            value_name = self._get_name(node.value)
            if value_name:
                return f"{value_name}.{node.attr}"
            return node.attr
        if isinstance(node, ast.Subscript):
            return self._get_name(node.value)
        if isinstance(node, ast.Call):
            return self._get_name(node.func)
        return ""

    def _get_decorator_name(self, node) -> str:
        if isinstance(node, ast.Call):
            return self._get_name(node.func)
        return self._get_name(node)


# ── Main Ingester ──


class ProjectIngester:
    """Analyzes a Python project: parses code, extracts dependencies, maps data flow.

    Works standalone or with a SessionManager. When used with SessionManager,
    pass ``session.project_copy_dir`` as ``project_path`` and
    ``session.environment_info.installed_packages`` as ``installed_packages``.

    Usage::

        ingester = ProjectIngester("/path/to/project")
        result = ingester.ingest()
        print(f"Analyzed {result.files_analyzed} files, found {len(result.dependencies)} deps")
    """

    def __init__(
        self,
        project_path: str | Path,
        installed_packages: Optional[dict[str, str]] = None,
        pypi_timeout: float = 5.0,
        skip_pypi_check: bool = False,
    ):
        self._project_path = Path(project_path).resolve()
        if not self._project_path.is_dir():
            raise IngestionError(
                f"Project path does not exist or is not a directory: {self._project_path}"
            )

        self._installed_packages = installed_packages
        self._pypi_timeout = pypi_timeout
        self._skip_pypi_check = skip_pypi_check

    # ── Public API ──

    def ingest(self) -> IngestionResult:
        """Analyze the project and return complete ingestion results."""
        result = IngestionResult(project_path=str(self._project_path))

        # 1. Discover Python files
        py_files = self._discover_python_files()
        if not py_files:
            result.warnings.append("No Python files found in project")
            return result

        # 2. Parse each file
        for file_path in py_files:
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

        # 7. Summary
        total = result.files_analyzed + result.files_failed
        if result.files_failed > 0:
            result.warnings.append(
                f"Analyzed {result.files_analyzed} of {total} files; "
                f"{result.files_failed} couldn't be parsed"
            )

        return result

    # ── File Discovery ──

    def _discover_python_files(self) -> list[Path]:
        """Find all .py files in the project, skipping non-project directories."""
        py_files = []
        for path in self._project_path.rglob("*.py"):
            parts = path.relative_to(self._project_path).parts
            skip = False
            for part in parts[:-1]:  # check parent directories
                if part in DISCOVER_EXCLUDE_DIRS or part.endswith(".egg-info") or part.startswith(DISCOVER_EXCLUDE_PREFIXES):
                    skip = True
                    break
            if not skip:
                py_files.append(path)
        return sorted(py_files)

    # ── File Parsing ──

    def _parse_file(self, file_path: Path, rel_path: str) -> FileAnalysis:
        """Parse a single Python file and extract its structure."""
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
            tree = ast.parse(source, filename=str(file_path))
        except SyntaxError as e:
            analysis.parse_error = f"Syntax error at line {e.lineno}: {e.msg}"
            analysis.imports = self._extract_imports_regex(source, rel_path)
            return analysis

        analyzer = _FileAnalyzer(rel_path)
        result = analyzer.analyze(tree)

        analysis.functions = result.functions
        analysis.classes = result.classes
        analysis.imports = result.imports
        analysis.global_vars = result.global_vars

        return analysis

    @staticmethod
    def _extract_imports_regex(source: str, rel_path: str) -> list[ImportInfo]:
        """Best-effort import extraction using regex when ast.parse fails.

        Handles the two standard forms:
            import foo
            import foo as bar
            import foo, bar
            from foo import bar
            from foo import bar, baz
            from foo import (bar, baz)
        """
        imports: list[ImportInfo] = []

        for lineno, line in enumerate(source.splitlines(), start=1):
            stripped = line.strip()

            # Skip comments and blanks
            if not stripped or stripped.startswith("#"):
                continue

            # "from X import Y" form
            m = re.match(
                r"^from\s+([\w.]+)\s+import\s+(.+?)(?:\s*#.*)?$", stripped
            )
            if m:
                module = m.group(1)
                names_part = m.group(2).strip().strip("()")
                names = [n.strip().split(" as ")[0].strip()
                         for n in names_part.split(",") if n.strip()]
                imports.append(ImportInfo(
                    module=module,
                    names=names,
                    is_from_import=True,
                    lineno=lineno,
                ))
                continue

            # "import X" form (possibly "import X as Y", "import X, Y")
            m = re.match(
                r"^import\s+(.+?)(?:\s*#.*)?$", stripped
            )
            if m:
                for part in m.group(1).split(","):
                    part = part.strip()
                    if not part:
                        continue
                    pieces = re.split(r"\s+as\s+", part, maxsplit=1)
                    module_name = pieces[0].strip()
                    alias = pieces[1].strip() if len(pieces) > 1 else None
                    imports.append(ImportInfo(
                        module=module_name,
                        alias=alias,
                        is_from_import=False,
                        lineno=lineno,
                    ))

        return imports

    # ── Dependency Extraction ──

    def _extract_dependencies(self) -> list[DependencyInfo]:
        """Extract declared dependencies from all dependency files in the project."""
        seen: dict[str, DependencyInfo] = {}  # normalized_name -> DependencyInfo

        parsers = [
            (
                self._parse_requirements_txt,
                [
                    "requirements.txt", "requirement.txt",
                    "requirements-dev.txt", "requirements_dev.txt",
                ],
            ),
            (self._parse_pyproject_toml, ["pyproject.toml"]),
            (self._parse_setup_py, ["setup.py"]),
            (self._parse_setup_cfg, ["setup.cfg"]),
        ]

        for parser, filenames in parsers:
            for filename in filenames:
                path = self._project_path / filename
                if path.is_file():
                    for name, version_spec in parser(path):
                        normalized = _normalize_package_name(name)
                        if normalized not in seen or (
                            version_spec and not seen[normalized].required_version
                        ):
                            seen[normalized] = DependencyInfo(
                                name=name,
                                required_version=version_spec or None,
                                source_file=filename,
                            )

        deps = list(seen.values())

        # Enrich with installed versions
        installed = self._detect_installed_versions([d.name for d in deps])
        for dep in deps:
            norm = _normalize_package_name(dep.name)
            for inst_name, inst_ver in installed.items():
                if _normalize_package_name(inst_name) == norm:
                    dep.installed_version = inst_ver
                    break
            else:
                dep.is_missing = True

        # Enrich with latest versions from PyPI
        latest = self._check_latest_versions([d.name for d in deps])
        for dep in deps:
            norm = _normalize_package_name(dep.name)
            for pypi_name, pypi_ver in latest.items():
                if _normalize_package_name(pypi_name) == norm:
                    dep.latest_version = pypi_ver
                    break

            if dep.installed_version and dep.latest_version:
                dep.is_outdated = _is_version_outdated(
                    dep.installed_version, dep.latest_version
                )

        return deps

    # ── Dependency File Parsers ──

    def _parse_requirements_txt(self, path: Path) -> list[tuple[str, str]]:
        """Parse requirements.txt, returning (name, version_spec) tuples."""
        deps: list[tuple[str, str]] = []
        try:
            content = _read_text_safe(path)
        except OSError as e:
            logger.warning("Failed to read %s: %s", path, e)
            return deps

        for line in content.splitlines():
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("-"):
                continue
            # Strip inline comments
            if " #" in line:
                line = line[: line.index(" #")].strip()
            name, version_spec = _parse_dep_string(line)
            if name:
                deps.append((name, version_spec))
        return deps

    def _parse_pyproject_toml(self, path: Path) -> list[tuple[str, str]]:
        """Parse pyproject.toml for [project] dependencies."""
        deps: list[tuple[str, str]] = []
        try:
            content = path.read_bytes()
        except OSError as e:
            logger.warning("Failed to read %s: %s", path, e)
            return deps

        # Try structured parsing first
        if tomllib is not None:
            try:
                data = tomllib.loads(content.decode("utf-8"))
                for dep_str in data.get("project", {}).get("dependencies", []):
                    name, version = _parse_dep_string(dep_str)
                    if name:
                        deps.append((name, version))
                return deps
            except Exception as e:
                logger.warning("Failed to parse %s with tomllib: %s", path, e)

        # Fallback: regex extraction
        text = content.decode("utf-8", errors="replace")
        match = re.search(
            r"\[project\].*?dependencies\s*=\s*\[(.*?)\]", text, re.DOTALL
        )
        if match:
            for line in match.group(1).splitlines():
                line = line.strip().strip(",").strip("\"'")
                if line and not line.startswith("#"):
                    name, version = _parse_dep_string(line)
                    if name:
                        deps.append((name, version))
        return deps

    def _parse_setup_py(self, path: Path) -> list[tuple[str, str]]:
        """Parse setup.py using AST to extract install_requires."""
        deps: list[tuple[str, str]] = []
        try:
            source = _read_text_safe(path)
            tree = ast.parse(source)
        except (OSError, SyntaxError) as e:
            logger.warning("Failed to parse %s: %s", path, e)
            return deps

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func_name = _get_static_call_name(node)
            if func_name not in ("setup", "setuptools.setup"):
                continue
            for kw in node.keywords:
                if kw.arg == "install_requires":
                    for s in _extract_string_list(kw.value):
                        name, version = _parse_dep_string(s)
                        if name:
                            deps.append((name, version))
                    break
        return deps

    def _parse_setup_cfg(self, path: Path) -> list[tuple[str, str]]:
        """Parse setup.cfg [options] install_requires."""
        deps: list[tuple[str, str]] = []
        config = configparser.ConfigParser()
        try:
            text = _read_text_safe(path)
            config.read_string(text, source=str(path))
        except (configparser.Error, OSError) as e:
            logger.warning("Failed to parse %s: %s", path, e)
            return deps

        if not config.has_option("options", "install_requires"):
            return deps

        requires_str = config.get("options", "install_requires")
        for line in requires_str.splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                name, version = _parse_dep_string(line)
                if name:
                    deps.append((name, version))
        return deps

    # ── Version Detection ──

    def _detect_installed_versions(self, dep_names: list[str]) -> dict[str, str]:
        """Get installed versions for the given dependency names."""
        versions: dict[str, str] = {}

        if self._installed_packages is not None:
            normalized_lookup = {
                _normalize_package_name(k): (k, v)
                for k, v in self._installed_packages.items()
            }
            for name in dep_names:
                norm = _normalize_package_name(name)
                if norm in normalized_lookup:
                    _, version = normalized_lookup[norm]
                    versions[name] = version
            return versions

        # Detect from current environment via importlib.metadata
        for name in dep_names:
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                # Try normalized name
                norm = _normalize_package_name(name).replace("-", "_")
                try:
                    versions[name] = importlib.metadata.version(norm)
                except importlib.metadata.PackageNotFoundError:
                    pass
        return versions

    # ── Transitive Dependency Resolution ──

    def _resolve_dependency_tree(self, root_deps: list[str]) -> dict[str, list[str]]:
        """Build transitive dependency tree from declared dependencies.

        Uses importlib.metadata to discover each package's own requirements,
        then recurses. Packages not installed in the current environment are
        recorded with an empty dependency list.
        """
        tree: dict[str, list[str]] = {}
        visited: set[str] = set()

        def _resolve(pkg_name: str):
            normalized = _normalize_package_name(pkg_name)
            if normalized in visited:
                return
            visited.add(normalized)

            try:
                reqs = importlib.metadata.requires(pkg_name)
            except importlib.metadata.PackageNotFoundError:
                tree[pkg_name] = []
                return

            if reqs is None:
                tree[pkg_name] = []
                return

            direct_deps = []
            for req_str in reqs:
                # Skip optional/extra dependencies
                if "extra ==" in req_str:
                    continue
                match = _PKG_NAME_RE.match(req_str.strip())
                if match:
                    dep_name = match.group(1)
                    direct_deps.append(dep_name)
                    _resolve(dep_name)

            tree[pkg_name] = direct_deps

        for dep in root_deps:
            _resolve(dep)

        return tree

    # ── PyPI Version Checking ──

    def _check_latest_versions(self, dep_names: list[str]) -> dict[str, str]:
        """Query PyPI for latest stable versions. Parallelized for speed."""
        if self._skip_pypi_check or not dep_names:
            return {}

        latest: dict[str, str] = {}
        workers = min(8, len(dep_names))

        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_name = {
                executor.submit(self._fetch_pypi_version, name): name
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

    def _fetch_pypi_version(self, package_name: str) -> Optional[str]:
        """Fetch the latest stable version of a package from PyPI."""
        url = f"https://pypi.org/pypi/{package_name}/json"
        try:
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=self._pypi_timeout) as resp:
                data = json.loads(resp.read())
                return data.get("info", {}).get("version")
        except (
            urllib.error.URLError,
            urllib.error.HTTPError,
            json.JSONDecodeError,
            OSError,
            TimeoutError,
        ) as e:
            logger.debug("Failed to fetch PyPI version for %s: %s", package_name, e)
            return None

    # ── Function Flow Mapping ──

    def _build_function_flows(
        self, file_analyses: list[FileAnalysis]
    ) -> list[FunctionFlow]:
        """Build call graph edges between functions across the project."""
        flows: list[FunctionFlow] = []

        # Build global function registry: qualified_name -> FunctionInfo
        func_registry: dict[str, FunctionInfo] = {}
        for analysis in file_analyses:
            module = self._file_to_module(analysis.file_path)
            for func in analysis.functions:
                if func.class_name:
                    qname = f"{module}.{func.class_name}.{func.name}"
                else:
                    qname = f"{module}.{func.name}"
                func_registry[qname] = func

        # For each file, resolve calls to qualified names
        for analysis in file_analyses:
            module = self._file_to_module(analysis.file_path)

            # Build import resolution map for this file
            import_map: dict[str, str] = {}
            for imp in analysis.imports:
                if imp.is_from_import:
                    for name in imp.names:
                        import_map[name] = f"{imp.module}.{name}"
                else:
                    key = imp.alias if imp.alias else imp.module
                    import_map[key] = imp.module

            # Local function lookup
            local_funcs: dict[str, str] = {}
            for func in analysis.functions:
                if func.class_name:
                    local_funcs[f"{func.class_name}.{func.name}"] = (
                        f"{module}.{func.class_name}.{func.name}"
                    )
                    # Also allow bare method name for self.method() -> Class.method
                    local_funcs[f"self.{func.name}"] = (
                        f"{module}.{func.class_name}.{func.name}"
                    )
                else:
                    local_funcs[func.name] = f"{module}.{func.name}"

            # Resolve calls in each function
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
        """Resolve a call name to a qualified name using local defs and imports."""
        # Check local functions first
        if call_name in local_funcs:
            return local_funcs[call_name]

        # Check imports
        parts = call_name.split(".")
        if parts[0] in import_map:
            resolved_base = import_map[parts[0]]
            if len(parts) > 1:
                return f"{resolved_base}.{'.'.join(parts[1:])}"
            return resolved_base

        # Unresolved — return as-is (likely a builtin or dynamic call)
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

        # 2. Cross-module dependency hubs (project-internal modules imported by many)
        project_modules = {
            self._file_to_module(a.file_path) for a in file_analyses
        }
        module_importers: dict[str, set[str]] = defaultdict(set)
        for analysis in file_analyses:
            src_module = self._file_to_module(analysis.file_path)
            for imp in analysis.imports:
                target = imp.module.split(".")[0] if imp.module else ""
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

        # 3. Shared mutable global state (via `global` keyword)
        global_accessors: dict[tuple[str, str], list[str]] = defaultdict(list)
        for analysis in file_analyses:
            module = self._file_to_module(analysis.file_path)
            for func in analysis.functions:
                for gname in func.globals_accessed:
                    prefix = f"{func.class_name}." if func.class_name else ""
                    qualified = f"{module}.{prefix}{func.name}"
                    global_accessors[(module, gname)].append(qualified)

        for (module, var), accessors in global_accessors.items():
            unique = sorted(set(accessors))
            if len(unique) >= 2:
                points.append(
                    CouplingPoint(
                        source=f"{module}.{var}",
                        targets=unique,
                        coupling_type="shared_state",
                        description=(
                            f"Global variable '{var}' in module '{module}' is "
                            f"mutated by {len(unique)} functions"
                        ),
                    )
                )

        return points

    # ── Helpers ──

    def _file_to_module(self, rel_path: str) -> str:
        """Convert a relative file path to a Python module name."""
        path = Path(rel_path)
        parts = list(path.parts)
        if parts[-1] == "__init__.py":
            parts = parts[:-1]
        else:
            parts[-1] = path.stem
        return ".".join(parts) if parts else ""


# ── Module-Level Helpers (used by setup.py parser) ──


def _get_static_call_name(node: ast.Call) -> str:
    """Get the function name from a Call node for setup.py parsing."""
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
        return f"{node.func.value.id}.{node.func.attr}"
    return ""


def _extract_string_list(node: ast.expr) -> list[str]:
    """Extract a list of string literals from an AST node."""
    strings = []
    if isinstance(node, ast.List):
        for elt in node.elts:
            if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                strings.append(elt.value)
    return strings
