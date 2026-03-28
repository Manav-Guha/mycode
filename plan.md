# Plan: Replace JS/TS Regex Ingester with TypeScript Compiler API Parser

## Problem

The JS ingester (`src/mycode/js_ingester.py`, class `_JsFileAnalyzer`) uses 12+ regex patterns to extract functions, imports, classes, and dependency usage from JS/TS files. This misses:

- **Nested functions** — regex only finds top-level and class methods; inner functions/closures invisible
- **Dynamic imports** — `import(variable)` partially handled but `import(template literal)` missed
- **TypeScript specifics** — generics with nested angle brackets break regex, complex type annotations confuse function extraction, interface/type/enum declarations ignored
- **Complex module patterns** — `export default class {}`, re-exports with renaming, `export =`, barrel files with `export * from`
- **JSX/TSX** — JSX expressions containing `{` confuse brace-depth tracking used for scope detection
- **Decorators** — TypeScript/experimental decorators not captured
- **Arrow functions without parens** — `x => x + 1` not matched by `_JS_ARROW_FUNC_RE` (requires `(`)

The Python ingester uses `ast.parse()` and gets full-fidelity results. The JS path needs parity.

---

## Current Architecture

### JS File Parsing (what we're replacing)
- `_JsFileAnalyzer.analyze(source)` in `js_ingester.py:422-456`
- Steps: strip comments → compute brace depths → extract classes → extract functions → extract imports → extract global vars
- Returns `FileAnalysis` (shared dataclass from `ingester.py:146`)
- Called by `JsProjectIngester._parse_file()` at line 858

### Existing Node.js Subprocess Pattern (what we're replicating)
- `js_module_loader.js` — Node.js script, reads JSON from stdin, writes JSON to stdout
- `js_module_loader.py` — Python wrapper: `_check_node_available()`, `subprocess.run()`, JSON parse, timeout handling, dataclass results
- Pattern: `task = json.dumps({...})` → `subprocess.run([node_path, script], input=task, ...)` → parse stdout JSON

### Data Structures (must match exactly)
The parser output must populate these existing dataclasses:
- `FunctionInfo(name, file_path, lineno, end_lineno, args, decorators, calls, is_method, class_name, is_async, globals_accessed)`
- `ClassInfo(name, file_path, lineno, end_lineno, methods, bases, decorators)`
- `ImportInfo(module, names, alias, is_from_import, lineno)`
- `GlobalVarInfo(name, file_path, lineno)`
- `FileAnalysis(file_path, functions, classes, imports, global_vars, parse_error, lines_of_code)`

### Export info (what `_JsFileAnalyzer` does NOT currently track)
- Whether a function/class is exported — not in the current dataclasses
- We won't add an `exported` field to `FunctionInfo`/`ClassInfo` yet, because nothing downstream consumes it and the acceptance criteria says "do not change report output format". We can capture it in the JSON output from the parser for future use, but silently drop it in the Python wrapper.

**Decision needed:** The acceptance criteria say "Outputs JSON with: ... export statements, ... dependency usage patterns (which imported modules are called where)." These are not in the current dataclasses. Options:
1. Include them in the parser JSON output but don't wire into dataclasses (future-proof, no downstream change)
2. Skip them entirely

**Recommendation:** Option 1 — the parser outputs them, the Python wrapper ignores them for now.

---

## Design

### New Files

#### 1. `src/mycode/js_parser.js` — TypeScript Compiler API parser

Single Node.js script. One dependency: `typescript` (via npm).

**Input** (JSON on stdin):
```json
{
  "command": "parse",
  "file_path": "/absolute/path/to/file.tsx",
  "source": "...optional, if provided parse this instead of reading file..."
}
```

**Output** (JSON on stdout):
```json
{
  "status": "ok",
  "functions": [
    {
      "name": "handleClick",
      "lineno": 10,
      "end_lineno": 25,
      "args": ["event", "options"],
      "is_async": false,
      "is_method": false,
      "class_name": null,
      "decorators": [],
      "calls": ["console.log", "setState", "fetchData"],
      "exported": false
    }
  ],
  "classes": [
    {
      "name": "UserService",
      "lineno": 30,
      "end_lineno": 80,
      "bases": ["BaseService"],
      "methods": ["getUser", "saveUser"],
      "decorators": ["Injectable"],
      "exported": true
    }
  ],
  "imports": [
    {
      "module": "react",
      "names": ["useState", "useEffect"],
      "alias": null,
      "is_from_import": true,
      "lineno": 1
    }
  ],
  "global_vars": [
    {"name": "API_URL", "lineno": 5}
  ],
  "exports": ["handleClick", "UserService", "API_URL"],
  "lines_of_code": 95
}
```

**Error output:**
```json
{
  "status": "error",
  "error": "description",
  "error_type": "ParseError"
}
```

**Implementation approach:**
- `ts.createSourceFile(fileName, source, ts.ScriptTarget.Latest, true)` — parses any JS/TS/JSX/TSX without needing tsconfig
- Walk the AST with `ts.forEachChild` recursively
- Track scope (class context, nesting) via a visitor stack
- For call extraction within function bodies: walk function body node, collect `ts.SyntaxKind.CallExpression` identifiers
- For decorators: check `ts.canHaveDecorators(node)` → `ts.getDecorators(node)`
- Line numbers: `sourceFile.getLineAndCharacterOfPosition(node.getStart())`

**Key design decisions:**
- Parse with `setParentNodes: true` so we can walk up to determine class membership
- Use `ts.ScriptTarget.Latest` + `ts.ScriptKind` auto-detected from extension to handle all four file types
- For syntax errors: TypeScript parser is error-recovering — it still produces an AST. We parse what we can and report diagnostics as warnings, not failures.
- The script does NOT require project-level `tsconfig.json` or `node_modules`. It's pure syntactic parsing, no type checking.
- Timeout: 10s per file (same as module loader)

#### 2. Python wrapper — modifications to `js_ingester.py`

**New function: `_parse_file_with_ts_parser()`**

```python
def _parse_file_with_ts_parser(
    file_path: str,
    rel_path: str,
    source: str,
    node_path: str = "node",
    timeout: int = 10,
) -> FileAnalysis | None:
    """Try to parse a JS/TS file using the TypeScript compiler API.

    Returns FileAnalysis on success, None on failure (caller falls back to regex).
    """
```

- Locates `js_parser.js` via `Path(__file__).parent / "js_parser.js"` (same pattern as `js_module_loader.py`)
- Checks Node.js availability (cache result for session — don't re-check per file)
- Checks `typescript` npm package availability (single check: `node -e "require('typescript')"`)
- Sends source via stdin JSON, reads stdout JSON
- Converts JSON output → `FileAnalysis` with proper `FunctionInfo`, `ClassInfo`, `ImportInfo`, `GlobalVarInfo`
- On any failure (Node not available, typescript not installed, parse error, timeout, bad JSON): returns `None`

**Modification to `JsProjectIngester._parse_file()`** (line 840-868):

```python
def _parse_file(self, file_path: Path, rel_path: str) -> FileAnalysis:
    # Read source (same as now)
    source = ...

    # Try AST parser first
    if self._use_ast_parser:
        ast_result = _parse_file_with_ts_parser(
            str(file_path), rel_path, source
        )
        if ast_result is not None:
            return ast_result

    # Fallback: existing regex analysis (unchanged)
    analyzer = _JsFileAnalyzer(rel_path)
    result = analyzer.analyze(source)
    ...
```

**New `__init__` parameter:** `use_ast_parser: bool = True` — allows disabling AST parser in tests or if Node.js unavailable. Auto-set to `False` on first failure to avoid repeated subprocess spawns.

**Caching strategy:**
- `_node_available: bool | None = None` — checked once per ingester instance
- `_ts_available: bool | None = None` — checked once per ingester instance
- If either is False, skip AST parser for all files (fall back to regex silently)

### npm Dependency: `typescript`

- The `typescript` package is needed at runtime when parsing JS/TS files
- It should be installed in myCode's own environment, NOT in the user's project
- Location: `src/mycode/node_modules/typescript` — installed during myCode's own setup
- Add to project's `package.json` (create one at `src/mycode/package.json` if needed) or document as a prerequisite
- **Alternative**: bundle with myCode's pip package — but npm packages can't go in pip wheels. Better: check on first use, log clear warning if missing, fall back to regex.

**Decision needed:** Where should `typescript` be installed?
- Option A: `src/mycode/package.json` with `npm install` as build step
- Option B: Expect it globally (`npm install -g typescript`) — fragile
- Option C: Auto-install into a known location on first use — magic, risky

**Recommendation:** Option A. Add `src/mycode/package.json` with `{"dependencies": {"typescript": "^5.x"}}`. Document `npm install` in setup. The parser script resolves `typescript` from its own `__dirname/node_modules`.

---

## Fallback Behavior

The regex ingester is **never deleted**. Fallback chain:

1. Check Node.js available → if no, use regex for all files, log once
2. Check `typescript` importable → if no, use regex for all files, log once
3. Per file: call `js_parser.js` → if it fails (timeout, bad output, crash), use regex for that file, log warning
4. If AST parser succeeds: use its result, skip regex entirely for that file

No user-visible change in behavior when fallback activates. The report looks the same — just with less accurate analysis.

---

## File Changes Summary

| File | Change |
|------|--------|
| `src/mycode/js_parser.js` | **NEW** — TypeScript Compiler API parser script |
| `src/mycode/package.json` | **NEW** — declares `typescript` dependency |
| `src/mycode/js_ingester.py` | **MODIFY** — add `_parse_file_with_ts_parser()`, modify `_parse_file()` and `__init__()` |
| `tests/test_js_parser.py` | **NEW** — tests for the AST parser (10+ tests) |
| `tests/test_js_ingester.py` | **MODIFY** — add tests for fallback behavior |

Files NOT modified:
- `ingester.py` (Python ingester — untouched)
- `report.py`, `documents.py` (report output — untouched)
- `js_module_loader.js/.py` (separate concern — untouched)
- Any existing test files (except `test_js_ingester.py` for fallback tests)

---

## Test Plan

### `tests/test_js_parser.py` — New (10+ tests)

1. **Plain JS function declarations** — function, arrow, expression, methods → correct name/args/lineno/calls
2. **TypeScript with type annotations** — generics, return types, parameter types don't break extraction
3. **JSX component** — React component with JSX return parsed correctly, not confused by `{` in JSX
4. **TSX component** — TypeScript + JSX combination
5. **ES6 imports** — default, named, namespace, side-effect, re-exports
6. **CommonJS require** — const, destructured, bare, dynamic
7. **Class declarations** — with extends, methods, decorators, static members
8. **Nested functions** — inner functions captured (improvement over regex)
9. **Empty file** — returns empty FileAnalysis, no crash
10. **Syntax errors** — malformed JS still produces partial results (TS parser is error-recovering)
11. **Dynamic imports** — `import()` with literal and variable arguments
12. **Export detection** — exported vs non-exported functions/classes distinguished in parser JSON
13. **Global variables** — module-level const/let/var captured
14. **Async functions** — async flag correctly set for async functions, async arrows, async methods

### `tests/test_js_ingester.py` — Additions

15. **Fallback when Node.js unavailable** — mock `shutil.which` to return None, verify regex path used
16. **Fallback when typescript not installed** — mock subprocess to fail on ts check, verify regex path used
17. **Fallback when parser crashes** — mock subprocess to return bad JSON, verify regex path used for that file
18. **AST parser caching** — verify Node/TS availability checked only once per ingester instance

### Existing tests
- All existing `test_js_ingester.py` tests must pass unchanged (they test regex behavior which remains as fallback)
- All 2,152+ fast tests must pass

---

## Execution Order

1. Create `src/mycode/package.json` and `npm install typescript`
2. Write `src/mycode/js_parser.js` — the Node.js parser script
3. Manually test parser script: `echo '{"command":"parse","file_path":"test.js"}' | node src/mycode/js_parser.js`
4. Write `_parse_file_with_ts_parser()` in `js_ingester.py`
5. Modify `JsProjectIngester.__init__()` and `_parse_file()` for AST-first-with-fallback
6. Write `tests/test_js_parser.py`
7. Add fallback tests to `tests/test_js_ingester.py`
8. Run full fast test suite, fix any failures
9. Test against a real JS project to verify end-to-end

---

## Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| `typescript` npm package adds ~40MB | It's a dev/analysis dependency, not shipped to user. Only needed on myCode's machine. |
| Subprocess spawn per file is slow | TypeScript parser is fast (~10ms per file). Total overhead for typical project (20-50 files) is ~1-2s. Acceptable. Could batch multiple files in one subprocess call if needed later. |
| Node.js not available on user's machine | Graceful fallback to regex. Log once. No crash. |
| TS compiler API changes between versions | Pin `typescript` version. TS compiler API is stable for basic AST walking. |
| Parser output doesn't match regex output exactly | Tests compare both paths on same input. Differences are improvements (nested functions found), not regressions. |

---

## Open Questions for Review

1. **Batch mode?** Should `js_parser.js` accept multiple files per invocation (JSONL) to reduce subprocess overhead? Adds complexity but could matter for large projects.
2. **`package.json` location** — `src/mycode/package.json` or project root? Root already has Python tooling; a nested one is cleaner but requires `npm install` in that directory.
3. **Export tracking** — capture in parser JSON and ignore in Python wrapper, or skip entirely?
4. **`call` extraction depth** — should nested calls inside callbacks/closures within a function body be attributed to the outer function? (Regex currently does this via brace range; AST can be more precise.)
