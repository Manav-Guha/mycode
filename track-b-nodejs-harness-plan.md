# Track B — Node.js Callable Harness Plan

## Architecture
- Node.js child process spawned from Python orchestrator
- Communication: JSON task via stdin, JSON results via stdout
- Zero npm dependencies in the harness — Node.js built-ins only (require/import, process.memoryUsage(), perf_hooks)

## Sessions

### B1: Module loader + export discovery
- CJS module loading via require() as primary path
- ESM module loading via dynamic import() as fallback
- TypeScript pre-compilation: compile .ts/.tsx to JS before loading
- Export discovery: load module, enumerate exported functions (name, arity, async), return as JSON
- Python-side wrapper: spawn Node process, send task, read result, parse JSON

### B2: JS synthetic data generator + instrumented caller
- Generate synthetic arguments for discovered functions based on parameter analysis
- Instrumented caller: invoke functions with timing, memory measurement, error capture
- Escalating data sizes matching Python harness pattern

### B3: Integration with execution engine
- Wire into engine.py _execute_scenario for JS callable scenarios
- Replace browser_framework skip with actual function-level testing for .js files
- .ts/.tsx/.jsx still skip unless B1 transpilation handles them

### B4: Validation against real repos + refinement
- Run against test portfolio JS repos (React Shopping Cart, Socket.io Chat, MERN Inventory)
- Compare findings against HTTP-only results
- Refine based on real-world edge cases

## Research confirmed
No existing tool does function-level JS stress testing. Artillery, Autocannon, k6 are all HTTP-level. The harness is built from scratch using Node.js primitives.
