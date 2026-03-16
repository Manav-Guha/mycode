#!/usr/bin/env node
"use strict";

/**
 * js_module_loader.js — Node.js module loader and export discovery.
 *
 * Spawned as a child process by the Python orchestrator.
 * Communication: JSON task on stdin, JSON result on stdout.
 * Zero npm dependencies — Node.js built-ins only.
 *
 * Task format:
 *   {"command": "discover", "file_path": "/absolute/path/to/module.js"}
 *
 * Result format (success):
 *   {"status": "ok", "exports": [...], "load_method": "cjs"|"esm"}
 *
 * Result format (error):
 *   {"status": "error", "error": "description", "error_type": "name"}
 */

const fs = require("fs");
const path = require("path");

function reply(obj) {
  process.stdout.write(JSON.stringify(obj) + "\n");
}

function isAsyncFunction(fn) {
  if (typeof fn !== "function") return false;
  const ctor = fn.constructor;
  if (ctor && ctor.name === "AsyncFunction") return true;
  // Also check the string representation for transpiled async functions
  const str = Function.prototype.toString.call(fn);
  return /^async\s/.test(str);
}

function enumerateExports(mod) {
  const exports = [];
  if (mod == null) return exports;

  // Collect named exports from the module object
  const seen = new Set();
  const keys = Object.keys(mod);
  for (const key of keys) {
    if (key === "__esModule" || key === "default") continue;
    if (seen.has(key)) continue;
    seen.add(key);

    const val = mod[key];
    if (typeof val === "function") {
      exports.push({
        name: key,
        arity: val.length,
        is_async: isAsyncFunction(val),
      });
    }
  }

  // Handle default export: ESM modules loaded via require() or import()
  // expose the default export as mod.default
  const def = mod.default;
  if (def != null && !seen.has("default")) {
    if (typeof def === "function") {
      // Default is a single function
      exports.push({
        name: def.name || "default",
        arity: def.length,
        is_async: isAsyncFunction(def),
      });
    } else if (typeof def === "object") {
      // Default is an object with function properties (e.g. export default { fn1, fn2 })
      for (const key of Object.keys(def)) {
        if (seen.has(key)) continue;
        seen.add(key);
        const val = def[key];
        if (typeof val === "function") {
          exports.push({
            name: key,
            arity: val.length,
            is_async: isAsyncFunction(val),
          });
        }
      }
    }
  }

  // If module.exports itself is a single function (CJS: module.exports = function)
  if (typeof mod === "function" && exports.length === 0) {
    exports.push({
      name: mod.name || "default",
      arity: mod.length,
      is_async: isAsyncFunction(mod),
    });
  }

  return exports;
}

async function loadModule(filePath) {
  // Try CJS first
  try {
    const resolved = require.resolve(filePath);
    const mod = require(resolved);
    return { mod, method: "cjs" };
  } catch (err) {
    const msg = String(err.message || err);
    const code = err.code || "";
    // If require fails because it's ESM, fall through to dynamic import
    const isEsmError =
      code === "ERR_REQUIRE_ESM" ||
      code === "ERR_REQUIRE_ASYNC_MODULE" ||
      /Cannot use import statement/i.test(msg) ||
      /Unexpected token 'export'/i.test(msg);

    if (!isEsmError) {
      // Genuine load failure — not a CJS/ESM issue
      throw err;
    }
  }

  // Fallback: dynamic import for ESM modules
  const fileUrl = "file://" + path.resolve(filePath);
  const mod = await import(fileUrl);
  return { mod, method: "esm" };
}

async function handleDiscover(filePath) {
  if (!filePath) {
    return reply({ status: "error", error: "file_path is required", error_type: "ValidationError" });
  }

  if (!fs.existsSync(filePath)) {
    return reply({ status: "error", error: "File not found: " + filePath, error_type: "FileNotFoundError" });
  }

  try {
    const { mod, method } = await loadModule(filePath);
    const exports = enumerateExports(mod);
    reply({ status: "ok", exports: exports, load_method: method });
  } catch (err) {
    const errType = (err && err.constructor) ? err.constructor.name : "Error";
    const errMsg = String(err.message || err).slice(0, 1000);
    reply({ status: "error", error: errMsg, error_type: errType });
  }
}

// Read task from stdin
let input = "";
process.stdin.setEncoding("utf8");
process.stdin.on("data", (chunk) => { input += chunk; });
process.stdin.on("end", async () => {
  let task;
  try {
    task = JSON.parse(input);
  } catch (err) {
    reply({ status: "error", error: "Invalid JSON on stdin", error_type: "ParseError" });
    return;
  }

  if (task.command === "discover") {
    await handleDiscover(task.file_path);
  } else {
    reply({ status: "error", error: "Unknown command: " + task.command, error_type: "ValidationError" });
  }
});
