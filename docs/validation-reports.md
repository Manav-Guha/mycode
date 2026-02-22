# myCode — Validation Reports

Anonymized results from stress testing real-world projects during development. These are actual myCode outputs, not synthetic examples.

---

## Project 1: Python Web App (Multi-Framework)

**Profile:** Flask + FastAPI + Streamlit + pandas + NumPy + requests + LangChain/ChromaDB
**Size:** 3 files, 829 lines, 14 declared dependencies
**Scenarios run:** 70
**Result:** 3 critical, 25 warnings, 21 clean

### Summary

```
We found some problems that could affect your project under real-world conditions.

- When handling multiple users at once, 1 request timed out.
- When downloading large responses, requests timed out under load.
- When calling external APIs that respond slowly, 1 request timed out.
```

### Key Degradation Curves

| Component | Metric | Start | End | Scaling Factor | Breaking Point |
|-----------|--------|-------|-----|----------------|----------------|
| Streamlit (table serialization) | Execution time | 0.42ms | 444ms | 1,058x | 10MB table |
| Streamlit (session data) | Memory | 0.08MB | 72MB | 901x | 100,000 rows |
| Flask (payload handling) | Execution time | 0.13ms | 770ms | 248x | 10MB payload |
| Flask (session writes) | Execution time | 0.16ms | 32ms | 199x | 1,000 writes |
| FastAPI (async handlers) | Execution time | 0.06ms | 75ms | 172x | 500 handlers |
| Coupling (data access) | Memory | 0.02MB | 26.5MB | 1,325x | 1,000 operations |

### Notable Findings
- 2 resource cap hits (timeout) on requests library tests — external HTTP calls with no timeout protection
- 12 missing dependencies flagged (declared in requirements but not installed in test environment)
- 9 unrecognized dependencies received generic stress testing
- pandas flagged as outdated (2.3.3 vs 3.0.1)

---

## Project 2: JavaScript Analysis Tool (React + Charting)

**Profile:** React + Chart.js + Plotly + date-fns + various UI libraries
**Size:** 17 files, 24,000+ lines
**Scenarios run:** 25
**Result:** 4 critical, 2 warnings, 16 clean

### Summary

```
We found some problems that could affect your project under real-world conditions.

- When receiving unexpected or unusual input, the code crashes instead of handling it gracefully.
- When running calculations across components, memory grows from 4MB to 43MB.
```

### Key Degradation Curves

| Component | Metric | Start | End | Scaling Factor | Breaking Point |
|-----------|--------|-------|-----|----------------|----------------|
| JSON.parse (coupling) | Execution time | 0.16ms | 76ms | 477x | 100,000 ops |
| JSON.stringify (coupling) | Execution time | 0.14ms | 53ms | 379x | 100,000 ops |
| fetch (coupling) | Execution time | 0.16ms | 40ms | 248x | 10,000 ops |
| fetch (coupling) | Memory | 4MB | 43.5MB | 10.8x | 100,000 ops |
| JSON.stringify (coupling) | Memory | 3.5MB | 31.7MB | 9.0x | 100,000 ops |

### Notable Findings
- 4 critical issues from edge case input handling — app crashes on malformed data instead of graceful error handling
- Coupling-heavy architecture: high fan-in on JSON operations and fetch calls
- Memory growth predictable and linear — no leaks, just scale
- 3 version discrepancies flagged

---

## Project 3: JavaScript Framework (React, Multi-Component)

**Profile:** React, multi-component architecture
**Size:** Multiple files, modular design
**Scenarios run:** 19
**Result:** 0 critical, 0 warnings, 19 clean

### Summary

```
Your project looks solid under the conditions we tested.

- When running calculations across components, memory grows from 3MB to 8MB.
- When running calculations across components, response time goes from instant to fast.
```

### Key Degradation Curves

| Component | Metric | Start | End | Scaling Factor | Breaking Point |
|-----------|--------|-------|-----|----------------|----------------|
| JSON.parse (coupling) | Execution time | 0.16ms | 76ms | 477x | 100,000 ops |
| JSON.stringify (coupling) | Execution time | 0.14ms | 53ms | 379x | 100,000 ops |
| Array operations (coupling) | Execution time | 0.10ms | 6.9ms | 69x | 10,000 ops |
| fetch (coupling) | Memory | 4MB | 43.5MB | 10.8x | 100,000 ops |

### Notable Findings
- All 19 scenarios passed cleanly — no crashes, no resource cap hits
- Degradation curves show predictable linear scaling, not exponential blowup
- Memory growth modest and proportional to data size
- Well-structured component architecture with manageable coupling

---

## Project 4: JavaScript Expense Tracker (React, Small)

**Profile:** React, single-page application
**Size:** Small project, minimal dependencies
**Scenarios run:** 5
**Result:** 0 critical, 1 warning, 4 clean

### Summary

```
Your project mostly handles stress well, but there are a few areas to watch.

- When running calculations across components, memory grows from 4MB to 17MB.
- When updating shared state, response time goes from 0.06ms to 0.99ms.
```

### Key Degradation Curves

| Component | Metric | Start | End | Scaling Factor | Breaking Point |
|-----------|--------|-------|-----|----------------|----------------|
| Coupling (general) | Memory | 4MB | 17MB | 4.8x | 100,000 ops |
| State updates | Execution time | 0.06ms | 0.99ms | 16.5x | high volume |

### Notable Findings
- 5 scenarios, 4 clean — minimal project with minimal risk surface
- 1 version discrepancy noted
- Memory growth modest and expected for React app

---

## Project 5: Python CLI Tool (Self-Test)

**Profile:** subprocess, json, pathlib, ast, urllib, signal, tempfile, os, re, collections
**Size:** 26 files, 24,053 lines
**Scenarios run:** 279
**Result:** 0 critical, 0 warnings, 279 clean

### Summary

```
Your project looks solid under the conditions we tested.

- When running calculations across components, memory grows from 0MB to 26MB.
- When running calculations across components, memory grows from 0MB to 116MB — if 10 users are active, that's 1162MB on your server.
- When running calculations across components, memory grows from 0MB to 8MB.
```

### Key Degradation Curves

| Component | Metric | Start | End | Scaling Factor | Breaking Point |
|-----------|--------|-------|-----|----------------|----------------|
| sorted (coupling) | Execution time | 0.01ms | 64ms | 6,421x | 1,000 ops |
| f.write_text (coupling) | Execution time | 0.19ms | 314ms | 1,653x | 100,000 ops |
| pathlib.Path (coupling) | Execution time | 0.20ms | 247ms | 1,237x | 1,000 ops |
| json.dumps (coupling) | Execution time | 0.78ms | 810ms | 1,038x | 1,000 ops |
| json.dumps (coupling) | Memory | 0.11MB | 116MB | 1,056x | 100,000 ops |
| urllib.request (coupling) | Memory | 0.06MB | 56MB | 941x | 10,000 ops |
| pathlib.Path (coupling) | Memory | 0.02MB | 26.5MB | 1,325x | 1,000 ops |

### Notable Findings
- All 279 scenarios passed — zero crashes, zero resource cap hits, zero errors
- Memory growth linear and proportional to data size — no leaks detected
- json.dumps at 116MB peak for 100,000 operations is the highest memory point; acceptable for a CLI tool running locally
- High scenario count (279) driven by coupling analysis generating scenarios for Python builtins (len, str, isinstance, range) — these are noise, not real risk. Scenario count scaling is a known post-launch improvement item
- myCode successfully stress-tested itself without any operational issues

---

## Summary Across All Validated Projects

| Project | Language | Files | Lines | Scenarios | Critical | Warnings | Clean |
|---------|----------|-------|-------|-----------|----------|----------|-------|
| Multi-framework web app | Python | 3 | 829 | 70 | 3 | 25 | 21 |
| Analysis tool | JavaScript | 17 | 24,000+ | 25 | 4 | 2 | 16 |
| Multi-component framework | JavaScript | Multiple | — | 19 | 0 | 0 | 19 |
| Expense tracker | JavaScript | Small | — | 5 | 0 | 1 | 4 |
| CLI tool (self-test) | Python | 26 | 24,053 | 279 | 0 | 0 | 279 |
| **Total** | | | | **398** | **7** | **28** | **339** |

398 scenarios executed across 5 real-world projects. 7 critical issues found, 28 warnings, 339 clean passes. All projects completed without myCode errors or crashes.

---

*myCode diagnoses — it does not prescribe. Interpret results in your context.*
