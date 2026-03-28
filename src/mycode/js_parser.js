#!/usr/bin/env node
"use strict";

/**
 * js_parser.js — TypeScript Compiler API parser for JS/TS/JSX/TSX files.
 *
 * Spawned as a child process by the Python orchestrator.
 * Communication: JSON on stdin, JSON on stdout.
 *
 * Input format (batch — array of files):
 *   [
 *     {"file_path": "/absolute/path.tsx", "source": "optional source text"},
 *     ...
 *   ]
 *
 * Output format:
 *   [
 *     {"status": "ok", "file_path": "...", "functions": [...], ...},
 *     {"status": "error", "file_path": "...", "error": "...", "error_type": "..."},
 *     ...
 *   ]
 *
 * Single dependency: typescript (resolved from __dirname/node_modules).
 */

const fs = require("fs");
const path = require("path");

// Resolve typescript from our own node_modules (installed via src/mycode/package.json)
let ts;
try {
  ts = require(path.join(__dirname, "node_modules", "typescript"));
} catch (_e) {
  // Fallback: try global/project typescript
  try {
    ts = require("typescript");
  } catch (_e2) {
    process.stdout.write(
      JSON.stringify([
        {
          status: "error",
          file_path: "",
          error: "typescript package not found",
          error_type: "ModuleNotFoundError",
        },
      ]) + "\n"
    );
    process.exit(0);
  }
}

// ── Helpers ──

function getScriptKind(filePath) {
  const ext = path.extname(filePath).toLowerCase();
  switch (ext) {
    case ".tsx":
      return ts.ScriptKind.TSX;
    case ".ts":
      return ts.ScriptKind.TS;
    case ".jsx":
      return ts.ScriptKind.JSX;
    case ".mjs":
    case ".cjs":
    case ".js":
    default:
      return ts.ScriptKind.JS;
  }
}

function getLineNumber(sourceFile, pos) {
  // 0-indexed line → 1-indexed
  return sourceFile.getLineAndCharacterOfPosition(pos).line + 1;
}

function getEndLineNumber(sourceFile, node) {
  return sourceFile.getLineAndCharacterOfPosition(node.getEnd()).line + 1;
}

function getNodeName(node) {
  if (node.name) {
    return node.name.text || node.name.escapedText || "";
  }
  return "";
}

// ── Call Extraction ──

function extractCalls(node, sourceFile, calls) {
  if (node.kind === ts.SyntaxKind.CallExpression) {
    const callName = expressionToString(node.expression);
    if (callName) {
      calls.push(callName);
    }
  }
  // Also catch tagged template expressions like html`...`
  if (node.kind === ts.SyntaxKind.TaggedTemplateExpression) {
    const tagName = expressionToString(node.tag);
    if (tagName) {
      calls.push(tagName);
    }
  }
  ts.forEachChild(node, (child) => extractCalls(child, sourceFile, calls));
}

function expressionToString(expr) {
  if (!expr) return null;
  if (expr.kind === ts.SyntaxKind.Identifier) {
    return expr.text || expr.escapedText || "";
  }
  if (expr.kind === ts.SyntaxKind.ThisKeyword) {
    return "this";
  }
  if (expr.kind === ts.SyntaxKind.PropertyAccessExpression) {
    const left = expressionToString(expr.expression);
    const right = expr.name ? expr.name.text || expr.name.escapedText : "";
    if (left && right) return left + "." + right;
    return right || left || null;
  }
  // For more complex expressions (element access, calls), skip
  return null;
}

// ── Decorator Extraction ──

function extractDecorators(node) {
  const decorators = [];
  // TS 5.x: ts.canHaveDecorators / ts.getDecorators
  if (ts.canHaveDecorators && ts.getDecorators) {
    const decs = ts.getDecorators(node);
    if (decs) {
      for (const d of decs) {
        const name = expressionToString(d.expression);
        if (name) decorators.push(name);
      }
    }
  }
  return decorators;
}

// ── Parameter Extraction ──

function extractParams(params) {
  const args = [];
  for (const p of params) {
    if (p.name.kind === ts.SyntaxKind.Identifier) {
      const name = p.name.text || p.name.escapedText || "";
      if (name) args.push(name);
    } else if (
      p.name.kind === ts.SyntaxKind.ObjectBindingPattern ||
      p.name.kind === ts.SyntaxKind.ArrayBindingPattern
    ) {
      // Destructured param — extract element names
      for (const el of p.name.elements) {
        if (el.kind === ts.SyntaxKind.OmittedExpression) continue;
        const elName = el.name || el.propertyName;
        if (elName && elName.kind === ts.SyntaxKind.Identifier) {
          args.push(elName.text || elName.escapedText || "");
        }
      }
    }
  }
  return args;
}

// ── JS Keywords (filter from calls) ──

const JS_KEYWORDS = new Set([
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
]);

// ── Main AST Walker ──

function parseFile(filePath, source) {
  const scriptKind = getScriptKind(filePath);
  const sourceFile = ts.createSourceFile(
    path.basename(filePath),
    source,
    ts.ScriptTarget.Latest,
    /* setParentNodes */ true,
    scriptKind
  );

  const functions = [];
  const classes = [];
  const imports = [];
  const globalVars = [];
  const exports = [];

  // Track which names are exported
  const exportedNames = new Set();

  function isExported(node) {
    if (!node.modifiers) return false;
    return node.modifiers.some(
      (m) =>
        m.kind === ts.SyntaxKind.ExportKeyword ||
        m.kind === ts.SyntaxKind.DefaultKeyword
    );
  }

  function hasExportKeyword(node) {
    if (!node.modifiers) return false;
    return node.modifiers.some((m) => m.kind === ts.SyntaxKind.ExportKeyword);
  }

  function isAsync(node) {
    if (!node.modifiers) return false;
    return node.modifiers.some((m) => m.kind === ts.SyntaxKind.AsyncKeyword);
  }

  // Collect calls within a function body, attributing nested calls to outer function
  function collectCalls(body) {
    if (!body) return [];
    const calls = [];
    extractCalls(body, sourceFile, calls);
    // Deduplicate and filter keywords
    const seen = new Set();
    const filtered = [];
    for (const c of calls) {
      const root = c.split(".")[0];
      if (!JS_KEYWORDS.has(root) && !seen.has(c)) {
        seen.add(c);
        filtered.push(c);
      }
    }
    return filtered;
  }

  // Check if a node is at module (top) level
  function isModuleLevel(node) {
    return node.parent && node.parent.kind === ts.SyntaxKind.SourceFile;
  }

  // Walk the AST
  function visit(node, className) {
    switch (node.kind) {
      // ── Import Declarations ──
      case ts.SyntaxKind.ImportDeclaration: {
        const moduleSpec = node.moduleSpecifier;
        const moduleName =
          moduleSpec && moduleSpec.text !== undefined ? moduleSpec.text : "";
        const names = [];
        let alias = null;
        const clause = node.importClause;

        if (clause) {
          // Default import
          if (clause.name) {
            names.push(clause.name.text);
          }
          // Named / namespace bindings
          const bindings = clause.namedBindings;
          if (bindings) {
            if (bindings.kind === ts.SyntaxKind.NamespaceImport) {
              // import * as X from ...
              alias = bindings.name.text;
            } else if (bindings.kind === ts.SyntaxKind.NamedImports) {
              for (const el of bindings.elements) {
                // Use the local name (propertyName is the original if renamed)
                names.push(el.name.text);
              }
            }
          }
        }

        imports.push({
          module: moduleName,
          names: names,
          alias: alias,
          is_from_import: true,
          lineno: getLineNumber(sourceFile, node.getStart()),
        });
        break;
      }

      // ── Export Declarations (re-exports) ──
      case ts.SyntaxKind.ExportDeclaration: {
        if (node.moduleSpecifier) {
          const moduleName = node.moduleSpecifier.text || "";
          const names = [];
          if (node.exportClause && node.exportClause.elements) {
            for (const el of node.exportClause.elements) {
              const name = el.name.text;
              names.push(name);
              exportedNames.add(name);
            }
          }
          // export * from 'module' — namespace re-export
          imports.push({
            module: moduleName,
            names: names,
            alias: null,
            is_from_import: true,
            lineno: getLineNumber(sourceFile, node.getStart()),
          });
        } else if (node.exportClause && node.exportClause.elements) {
          // export { a, b } — local re-exports
          for (const el of node.exportClause.elements) {
            exportedNames.add(
              (el.propertyName || el.name).text
            );
          }
        }
        break;
      }

      // ── Export Assignment: export = X or export default X ──
      case ts.SyntaxKind.ExportAssignment: {
        if (node.expression && node.expression.kind === ts.SyntaxKind.Identifier) {
          exportedNames.add(node.expression.text);
        }
        break;
      }

      // ── Function Declarations ──
      case ts.SyntaxKind.FunctionDeclaration: {
        const name = getNodeName(node);
        if (!name) break; // anonymous default export function — skip
        const exp = hasExportKeyword(node);
        if (exp) exportedNames.add(name);

        functions.push({
          name: name,
          lineno: getLineNumber(sourceFile, node.getStart()),
          end_lineno: getEndLineNumber(sourceFile, node),
          args: extractParams(node.parameters),
          is_async: isAsync(node),
          is_method: !!className,
          class_name: className || null,
          decorators: extractDecorators(node),
          calls: collectCalls(node.body),
          exported: exp,
        });
        break;
      }

      // ── Class Declarations ──
      case ts.SyntaxKind.ClassDeclaration: {
        const name = getNodeName(node);
        if (!name) break;
        const exp = hasExportKeyword(node);
        if (exp) exportedNames.add(name);

        const bases = [];
        if (node.heritageClauses) {
          for (const hc of node.heritageClauses) {
            if (hc.token === ts.SyntaxKind.ExtendsKeyword) {
              for (const t of hc.types) {
                const baseName = expressionToString(t.expression);
                if (baseName) bases.push(baseName);
              }
            }
          }
        }

        const methods = [];
        const classDecorators = extractDecorators(node);

        // Visit class members
        for (const member of node.members) {
          if (
            member.kind === ts.SyntaxKind.MethodDeclaration ||
            member.kind === ts.SyntaxKind.GetAccessor ||
            member.kind === ts.SyntaxKind.SetAccessor
          ) {
            const mName = getNodeName(member);
            if (!mName) continue;
            methods.push(mName);

            functions.push({
              name: mName,
              lineno: getLineNumber(sourceFile, member.getStart()),
              end_lineno: getEndLineNumber(sourceFile, member),
              args: extractParams(member.parameters),
              is_async: isAsync(member),
              is_method: true,
              class_name: name,
              decorators: extractDecorators(member),
              calls: collectCalls(member.body),
              exported: false,
            });
          } else if (member.kind === ts.SyntaxKind.Constructor) {
            functions.push({
              name: "constructor",
              lineno: getLineNumber(sourceFile, member.getStart()),
              end_lineno: getEndLineNumber(sourceFile, member),
              args: extractParams(member.parameters),
              is_async: false,
              is_method: true,
              class_name: name,
              decorators: [],
              calls: collectCalls(member.body),
              exported: false,
            });
            methods.push("constructor");
          }
        }

        classes.push({
          name: name,
          lineno: getLineNumber(sourceFile, node.getStart()),
          end_lineno: getEndLineNumber(sourceFile, node),
          bases: bases,
          methods: methods,
          decorators: classDecorators,
          exported: exp,
        });

        // Don't recurse into class body again — we handled members above
        return;
      }

      // ── Variable Statements (arrows, function expressions, globals) ──
      case ts.SyntaxKind.VariableStatement: {
        const isExp = hasExportKeyword(node);
        const decls = node.declarationList
          ? node.declarationList.declarations
          : [];

        for (const decl of decls) {
          const name = getNodeName(decl);
          if (!name) continue;
          if (isExp) exportedNames.add(name);

          const init = decl.initializer;
          const atModuleLevel = isModuleLevel(node);

          if (!init) {
            // No initializer — it's a global variable declaration
            if (atModuleLevel) {
              globalVars.push({
                name: name,
                lineno: getLineNumber(sourceFile, decl.getStart()),
              });
            }
            continue;
          }

          // Arrow function or function expression
          if (
            init.kind === ts.SyntaxKind.ArrowFunction ||
            init.kind === ts.SyntaxKind.FunctionExpression
          ) {
            functions.push({
              name: name,
              lineno: getLineNumber(sourceFile, node.getStart()),
              end_lineno: getEndLineNumber(sourceFile, init),
              args: extractParams(init.parameters),
              is_async: isAsync(init),
              is_method: !!className,
              class_name: className || null,
              decorators: [],
              calls: collectCalls(init.body),
              exported: isExp,
            });
          } else {
            // Other variable — global if at module level
            if (atModuleLevel) {
              globalVars.push({
                name: name,
                lineno: getLineNumber(sourceFile, decl.getStart()),
              });
            }
          }
        }
        // Don't recurse — we handled the declarations
        return;
      }

      // ── CommonJS require() patterns ──
      case ts.SyntaxKind.ExpressionStatement: {
        // Handle: require('module') as a side-effect import
        const expr = node.expression;
        if (
          expr &&
          expr.kind === ts.SyntaxKind.CallExpression &&
          expr.expression &&
          expr.expression.kind === ts.SyntaxKind.Identifier &&
          expr.expression.text === "require" &&
          expr.arguments.length === 1
        ) {
          const arg = expr.arguments[0];
          if (arg.kind === ts.SyntaxKind.StringLiteral) {
            imports.push({
              module: arg.text,
              names: [],
              alias: null,
              is_from_import: false,
              lineno: getLineNumber(sourceFile, node.getStart()),
            });
          } else {
            const exprText = source
              .slice(arg.getStart(), arg.getEnd())
              .slice(0, 60);
            imports.push({
              module: "<dynamic: " + exprText + ">",
              names: [],
              alias: null,
              is_from_import: false,
              lineno: getLineNumber(sourceFile, node.getStart()),
            });
          }
        }

        // Handle: module.exports = { ... } — track exported names
        if (
          expr &&
          expr.kind === ts.SyntaxKind.BinaryExpression &&
          expr.operatorToken.kind === ts.SyntaxKind.EqualsToken
        ) {
          const left = expressionToString(expr.left);
          if (left === "module.exports") {
            // module.exports = { a, b } or module.exports = X
            if (
              expr.right.kind === ts.SyntaxKind.ObjectLiteralExpression
            ) {
              for (const prop of expr.right.properties) {
                const pName = getNodeName(prop);
                if (pName) exportedNames.add(pName);
              }
            } else if (
              expr.right.kind === ts.SyntaxKind.Identifier
            ) {
              exportedNames.add(expr.right.text);
            }
          } else if (left && left.startsWith("module.exports.")) {
            exportedNames.add(left.split(".").pop());
          } else if (left && left.startsWith("exports.")) {
            exportedNames.add(left.split(".")[1]);
          }
        }
        break;
      }
    }

    // Recurse into children (except class bodies which we handle above)
    ts.forEachChild(node, (child) => visit(child, className));
  }

  // Handle top-level require() in variable declarations
  // We need a second pass because VariableStatement handling above
  // skips recursion. Scan for require patterns in variable inits.
  function scanRequires(node) {
    if (node.kind === ts.SyntaxKind.VariableStatement) {
      const decls = node.declarationList
        ? node.declarationList.declarations
        : [];
      for (const decl of decls) {
        const init = decl.initializer;
        if (!init) continue;

        // const X = require('module')
        if (
          init.kind === ts.SyntaxKind.CallExpression &&
          init.expression &&
          init.expression.kind === ts.SyntaxKind.Identifier &&
          init.expression.text === "require" &&
          init.arguments.length === 1
        ) {
          const arg = init.arguments[0];
          const names = [];
          const varName = getNodeName(decl);

          if (arg.kind === ts.SyntaxKind.StringLiteral) {
            // const X = require('module') — X is the name
            if (decl.name.kind === ts.SyntaxKind.Identifier) {
              names.push(varName);
            } else if (
              decl.name.kind === ts.SyntaxKind.ObjectBindingPattern
            ) {
              // const { a, b } = require('module')
              for (const el of decl.name.elements) {
                if (
                  el.name &&
                  el.name.kind === ts.SyntaxKind.Identifier
                ) {
                  names.push(el.name.text);
                }
              }
            }

            imports.push({
              module: arg.text,
              names: names,
              alias: null,
              is_from_import: false,
              lineno: getLineNumber(sourceFile, node.getStart()),
            });
          } else {
            // Dynamic require
            const exprText = source
              .slice(arg.getStart(), arg.getEnd())
              .slice(0, 60);
            imports.push({
              module: "<dynamic: " + exprText + ">",
              names: [],
              alias: null,
              is_from_import: false,
              lineno: getLineNumber(sourceFile, node.getStart()),
            });
          }
        }
      }
    }
    ts.forEachChild(node, scanRequires);
  }

  // Also scan for dynamic import() expressions anywhere
  function scanDynamicImports(node) {
    if (node.kind === ts.SyntaxKind.CallExpression) {
      const expr = node.expression;
      // import() expressions — expr.kind is ImportKeyword in some TS versions
      if (
        expr &&
        (expr.kind === ts.SyntaxKind.ImportKeyword ||
          (expr.kind === ts.SyntaxKind.Identifier && expr.text === "import"))
      ) {
        if (node.arguments && node.arguments.length === 1) {
          const arg = node.arguments[0];
          if (arg.kind === ts.SyntaxKind.StringLiteral) {
            imports.push({
              module: arg.text,
              names: [],
              alias: null,
              is_from_import: true,
              lineno: getLineNumber(sourceFile, node.getStart()),
            });
          } else {
            const exprText = source
              .slice(arg.getStart(), arg.getEnd())
              .slice(0, 60);
            imports.push({
              module: "<dynamic: " + exprText + ">",
              names: [],
              alias: null,
              is_from_import: true,
              lineno: getLineNumber(sourceFile, node.getStart()),
            });
          }
        }
      }
    }
    ts.forEachChild(node, scanDynamicImports);
  }

  // Run all passes
  ts.forEachChild(sourceFile, (child) => visit(child, null));
  scanRequires(sourceFile);
  scanDynamicImports(sourceFile);

  // Build exports list from tracked names
  const exportsList = Array.from(exportedNames);

  // Deduplicate imports by (module, lineno)
  const seenImports = new Set();
  const dedupedImports = [];
  for (const imp of imports) {
    const key = imp.module + ":" + imp.lineno;
    if (!seenImports.has(key)) {
      seenImports.add(key);
      dedupedImports.push(imp);
    }
  }

  return {
    status: "ok",
    file_path: filePath,
    functions: functions,
    classes: classes,
    imports: dedupedImports,
    global_vars: globalVars,
    exports: exportsList,
    lines_of_code: source.length === 0 ? 0 : source.split("\n").length,
  };
}

// ── Entry Point ──

let input = "";
process.stdin.setEncoding("utf8");
process.stdin.on("data", (chunk) => {
  input += chunk;
});
process.stdin.on("end", () => {
  let tasks;
  try {
    tasks = JSON.parse(input);
  } catch (_e) {
    process.stdout.write(
      JSON.stringify([
        {
          status: "error",
          file_path: "",
          error: "Invalid JSON on stdin",
          error_type: "ParseError",
        },
      ]) + "\n"
    );
    return;
  }

  // Normalize: accept single object or array
  if (!Array.isArray(tasks)) {
    tasks = [tasks];
  }

  const results = [];

  for (const task of tasks) {
    const filePath = task.file_path || "";

    try {
      // Get source: from task, or read from disk
      let source = task.source;
      if (source === undefined || source === null) {
        if (!filePath) {
          results.push({
            status: "error",
            file_path: filePath,
            error: "file_path is required when source is not provided",
            error_type: "ValidationError",
          });
          continue;
        }
        if (!fs.existsSync(filePath)) {
          results.push({
            status: "error",
            file_path: filePath,
            error: "File not found: " + filePath,
            error_type: "FileNotFoundError",
          });
          continue;
        }
        source = fs.readFileSync(filePath, "utf8");
      }

      const result = parseFile(filePath, source);
      results.push(result);
    } catch (err) {
      const errType =
        err && err.constructor ? err.constructor.name : "Error";
      const errMsg = String(err.message || err).slice(0, 1000);
      results.push({
        status: "error",
        file_path: filePath,
        error: errMsg,
        error_type: errType,
      });
    }
  }

  process.stdout.write(JSON.stringify(results) + "\n");
});
