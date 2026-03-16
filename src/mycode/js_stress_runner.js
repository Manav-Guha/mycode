#!/usr/bin/env node
"use strict";

/**
 * js_stress_runner.js — Instrumented JS function caller with synthetic data.
 *
 * Spawned as a child process by the Python orchestrator.
 * Communication: JSON task on stdin, structured results on stdout.
 * Zero npm dependencies — Node.js built-ins only.
 *
 * Task format:
 *   {
 *     "command": "stress",
 *     "file_path": "/absolute/path/to/module.js",
 *     "functions": [{"name": "fn", "arity": 2, "is_async": false}],
 *     "scale_levels": [100, 1000, 10000],
 *     "step_timeout_ms": 60000,
 *     "param_names": {"fn": ["data", "options"]}
 *   }
 *
 * Output: __MYCODE_RESULTS_START__ JSON __MYCODE_RESULTS_END__
 */

const fs = require("fs");
const path = require("path");

// ── Module Loading (shared with js_module_loader.js) ──

function isAsyncFunction(fn) {
  if (typeof fn !== "function") return false;
  const ctor = fn.constructor;
  if (ctor && ctor.name === "AsyncFunction") return true;
  return /^async\s/.test(Function.prototype.toString.call(fn));
}

async function loadModule(filePath) {
  try {
    const resolved = require.resolve(filePath);
    return require(resolved);
  } catch (err) {
    const msg = String(err.message || err);
    const code = err.code || "";
    const isEsmError =
      code === "ERR_REQUIRE_ESM" ||
      code === "ERR_REQUIRE_ASYNC_MODULE" ||
      /Cannot use import statement/i.test(msg) ||
      /Unexpected token 'export'/i.test(msg);
    if (!isEsmError) throw err;
  }
  const fileUrl = "file://" + path.resolve(filePath);
  return await import(fileUrl);
}

// ── Synthetic Data Generation ──

const _ID_PATTERN = /^(id|user_?id|item_?id|index|idx|num|number)$/i;
const _NAME_PATTERN = /^(name|label|title|key|tag|slug|username)$/i;
const _DATA_PATTERN = /^(data|input|items|list|array|rows|records|entries|values|payload|batch|collection)$/i;
const _CONFIG_PATTERN = /^(config|options|opts|settings|params|context|env|meta)$/i;
const _CALLBACK_PATTERN = /^(callback|cb|fn|handler|onComplete|onError|next|done|resolve|reject)$/i;
const _BOOL_PATTERN = /^(flag|enabled|active|is[A-Z_]|has[A-Z_]|should|verbose|force|dry_?run|debug)$/i;
const _COUNT_PATTERN = /^(count|size|limit|num|amount|length|max|min|offset|page|per_?page|depth|width|height|timeout|retries|port)$/i;
const _PATH_PATTERN = /^(path|url|uri|file|filename|filepath|dir|directory|endpoint|route|href|src|dest|source|target)$/i;
const _STRING_PATTERN = /^(text|message|description|content|body|query|search|filter|email|password|token|code|type|format|mode|status|role|category)$/i;

function generateArg(paramName, scaleLevel) {
  if (!paramName) return generatePositionalArg(0, scaleLevel);

  if (_CALLBACK_PATTERN.test(paramName)) return function() {};
  if (_BOOL_PATTERN.test(paramName)) return true;
  if (_ID_PATTERN.test(paramName)) return scaleLevel;
  if (_COUNT_PATTERN.test(paramName)) return scaleLevel;
  if (_PATH_PATTERN.test(paramName)) return "/test/path/" + scaleLevel;
  if (_NAME_PATTERN.test(paramName)) return "test_item_" + scaleLevel;
  if (_STRING_PATTERN.test(paramName)) return "x".repeat(Math.min(scaleLevel, 10000));
  if (_CONFIG_PATTERN.test(paramName)) {
    return { debug: false, timeout: scaleLevel, retries: 3, mode: "test", batch: scaleLevel };
  }
  if (_DATA_PATTERN.test(paramName)) return generateDataArray(scaleLevel);

  // Fallback: treat unknown names as data at higher arities, string otherwise
  return generatePositionalArg(0, scaleLevel);
}

function generateDataArray(size) {
  const arr = new Array(size);
  for (let i = 0; i < size; i++) {
    arr[i] = { id: i, value: Math.random() * 1000, label: "item_" + i };
  }
  return arr;
}

function generatePositionalArg(position, scaleLevel) {
  switch (position) {
    case 0: return generateDataArray(scaleLevel);
    case 1: return { debug: false, timeout: scaleLevel, retries: 3, mode: "test", batch: scaleLevel };
    case 2: return "test_string_" + scaleLevel;
    default: return null;
  }
}

function generateArgs(funcEntry, scaleLevel, paramNames) {
  const arity = funcEntry.arity;
  if (arity === 0) return [];

  const names = paramNames || [];
  const args = [];
  for (let i = 0; i < arity; i++) {
    if (i < names.length && names[i]) {
      args.push(generateArg(names[i], scaleLevel));
    } else {
      args.push(generatePositionalArg(i, scaleLevel));
    }
  }
  return args;
}

// ── Context Error Detection ──

const _CONTEXT_ERROR_PATTERNS = [
  /ECONNREFUSED/i,
  /ENOTFOUND/i,
  /ECONNRESET/i,
  /connect ETIMEDOUT/i,
  /getaddrinfo/i,
  /is not defined/i,
  /Missing required/i,
  /API.?key/i,
  /OPENAI_API_KEY/i,
  /ANTHROPIC_API_KEY/i,
  /DATABASE_URL/i,
  /not initialized/i,
  /app.?context/i,
  /application context/i,
  /no current event loop/i,
  /working outside/i,
];

const _CONTEXT_ERROR_TYPES = new Set([
  "ReferenceError",
  "TypeError",
]);

function isContextError(err) {
  if (err == null) return false;
  const errType = (err && err.constructor) ? err.constructor.name : "Error";
  const msg = String(err.message || err);

  // Module-not-found is always context
  if (err && err.code === "MODULE_NOT_FOUND") return true;

  // Network errors are context
  if (err && (err.code === "ECONNREFUSED" || err.code === "ENOTFOUND" || err.code === "ECONNRESET")) return true;

  // Check message patterns
  for (const pattern of _CONTEXT_ERROR_PATTERNS) {
    if (pattern.test(msg)) return true;
  }

  return false;
}

// ── Instrumented Execution ──

async function invokeWithTimeout(fn, args, timeoutMs) {
  return new Promise((resolve, reject) => {
    let timer = null;
    let settled = false;

    timer = setTimeout(() => {
      if (!settled) {
        settled = true;
        reject(new Error("Step timed out after " + timeoutMs + "ms"));
      }
    }, timeoutMs);

    try {
      const result = fn(...args);
      if (result && typeof result.then === "function") {
        result.then(
          (val) => { if (!settled) { settled = true; clearTimeout(timer); resolve(val); } },
          (err) => { if (!settled) { settled = true; clearTimeout(timer); reject(err); } }
        );
      } else {
        if (!settled) {
          settled = true;
          clearTimeout(timer);
          resolve(result);
        }
      }
    } catch (err) {
      if (!settled) {
        settled = true;
        clearTimeout(timer);
        reject(err);
      }
    }
  });
}

async function probeFunction(funcEntry, mod) {
  const fn = mod[funcEntry.name];
  if (typeof fn !== "function") {
    return { name: funcEntry.name, error: { type: "TypeError", message: funcEntry.name + " is not a function" } };
  }

  // Call with minimal safe args (not null — avoid false TypeError on null.property)
  const minArgs = [];
  for (let i = 0; i < funcEntry.arity; i++) {
    // First arg: empty array (most common data param), rest: empty object
    minArgs.push(i === 0 ? [] : {});
  }
  try {
    await invokeWithTimeout(fn, minArgs, 5000);
    return null; // probe passed
  } catch (err) {
    if (err != null && isContextError(err)) {
      const errType = (err.constructor) ? err.constructor.name : "Error";
      return {
        name: funcEntry.name,
        error: { type: errType, message: String(err.message || err).slice(0, 500) },
      };
    }
    // Non-context error (TypeError from bad args, etc.) — function is testable
    return null;
  }
}

async function measureStep(stepName, params, fn, stepTimeoutMs) {
  const errors = [];
  let capHit = "";
  const memBefore = process.memoryUsage();
  const t0 = performance.now();

  try {
    await invokeWithTimeout(fn, [], stepTimeoutMs);
  } catch (err) {
    const errType = (err != null && err.constructor) ? err.constructor.name : "Error";
    const errMsg = (err != null) ? String(err.message || err).slice(0, 500) : "Unknown error";

    if (/timed out/i.test(errMsg)) {
      capHit = "timeout";
    } else if (err instanceof RangeError && /call stack|Maximum/i.test(errMsg)) {
      capHit = "memory";
    }

    errors.push({
      type: errType,
      message: errMsg,
      traceback: (err && err.stack) ? err.stack.slice(-1000) : "",
    });
  }

  const elapsed = performance.now() - t0;
  const memAfter = process.memoryUsage();
  const peakMb = Math.max(memAfter.heapUsed, memBefore.heapUsed) / 1048576;

  return {
    step_name: stepName,
    parameters: params,
    execution_time_ms: Math.round(elapsed * 100) / 100,
    memory_peak_mb: Math.round(peakMb * 100) / 100,
    error_count: errors.length,
    errors: errors,
    resource_cap_hit: capHit,
  };
}

// ── Main Handler ──

async function handleStress(task) {
  const filePath = task.file_path;
  const functions = task.functions || [];
  const scaleLevels = task.scale_levels || [100, 1000, 10000];
  const stepTimeoutMs = task.step_timeout_ms || 60000;
  const paramNamesMap = task.param_names || {};

  const steps = [];
  const importErrors = [];
  const probeSkipped = [];

  // Load module
  let mod;
  try {
    mod = await loadModule(filePath);
  } catch (err) {
    const errType = (err && err.constructor) ? err.constructor.name : "Error";
    importErrors.push({
      type: errType,
      message: String(err.message || err).slice(0, 500),
      module: filePath,
    });
    return output({ steps, import_errors: importErrors, probe_skipped: probeSkipped });
  }

  // Unwrap ESM default if needed
  if (mod && mod.__esModule && mod.default && typeof mod.default === "object") {
    mod = Object.assign({}, mod.default, mod);
  }

  // Resolve functions to actual callables
  const resolved = [];
  for (const entry of functions) {
    const fn = mod[entry.name];
    if (typeof fn === "function") {
      resolved.push({
        name: entry.name,
        arity: entry.arity,
        is_async: entry.is_async || isAsyncFunction(fn),
        func: fn,
      });
    }
  }

  if (resolved.length === 0 && importErrors.length === 0) {
    // No callable functions found — nothing to stress
    return output({ steps, import_errors: importErrors, probe_skipped: probeSkipped });
  }

  // Probe each function
  const testable = [];
  for (const entry of resolved) {
    const probeResult = await probeFunction(entry, mod);
    if (probeResult) {
      probeSkipped.push(probeResult);
    } else {
      testable.push(entry);
    }
  }

  if (testable.length === 0) {
    return output({ steps, import_errors: importErrors, probe_skipped: probeSkipped });
  }

  // Run stress steps at each scale level
  const totalStartMs = performance.now();
  for (const level of scaleLevels) {
    // Time budget: skip remaining levels if 80% of total budget used
    const elapsedTotal = performance.now() - totalStartMs;
    const totalBudget = stepTimeoutMs * scaleLevels.length;
    if (elapsedTotal > totalBudget * 0.8 && steps.length > 0) {
      steps.push({
        step_name: "scale_" + level,
        parameters: { scale_level: level, functions_called: 0 },
        execution_time_ms: 0,
        memory_peak_mb: 0,
        error_count: 1,
        errors: [{ type: "Skipped", message: "Time budget exceeded (80%)", traceback: "" }],
        resource_cap_hit: "timeout",
      });
      continue;
    }

    const step = await measureStep(
      "scale_" + level,
      { scale_level: level, functions_called: testable.length },
      async () => {
        for (const entry of testable) {
          const args = generateArgs(entry, level, paramNamesMap[entry.name]);
          const fn = entry.func;
          const result = fn(...args);
          if (result && typeof result.then === "function") {
            await result;
          }
        }
      },
      stepTimeoutMs,
    );
    steps.push(step);

    // Stop escalating if step hit resource cap
    if (step.resource_cap_hit === "memory" || step.resource_cap_hit === "timeout") {
      break;
    }
  }

  output({ steps, import_errors: importErrors, probe_skipped: probeSkipped });
}

function output(data) {
  console.log("__MYCODE_RESULTS_START__");
  console.log(JSON.stringify(data));
  console.log("__MYCODE_RESULTS_END__");
}

// ── stdin reader ──

let input = "";
process.stdin.setEncoding("utf8");
process.stdin.on("data", (chunk) => { input += chunk; });
process.stdin.on("end", async () => {
  let task;
  try {
    task = JSON.parse(input);
  } catch (err) {
    console.error("Invalid JSON on stdin");
    process.exitCode = 1;
    return;
  }

  if (task.command === "stress") {
    try {
      await handleStress(task);
    } catch (err) {
      console.error("Fatal error:", err);
      output({ steps: [], import_errors: [], probe_skipped: [] });
      process.exitCode = 1;
    }
  } else {
    console.error("Unknown command: " + task.command);
    process.exitCode = 1;
  }
});
