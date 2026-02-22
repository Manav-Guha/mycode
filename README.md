# myCode

**Built it with AI? Test it before it breaks.**

Stress-testing tool for AI-generated code. Point it at your project, describe what it does, and find out where it breaks under real-world conditions.

## What It Does

You built an app with ChatGPT, Cursor, Claude, or Copilot. It works on your machine. But what happens when 100 users hit it at once? When the CSV grows to 100,000 rows? When someone uploads a 50MB file? When the API you depend on goes slow?

myCode answers these questions. It reads your code, understands your dependencies, generates stress scenarios, runs your actual code under escalating conditions, and tells you — in plain language — where it breaks.

## Quick Start

```bash
pip install mycode
mycode ~/path/to/your/project
```

That's it. myCode will:
1. Ask you what your project does (in your words, not engineering language)
2. Show you what it plans to test and let you approve
3. Run stress tests in a safe sandbox (your original files are never touched)
4. Give you a diagnostic report

## What It Tests

- **Data volume scaling** — what happens as your data grows from 100 rows to 100,000
- **Memory pressure** — does your app leak memory over repeated use
- **Concurrent load** — multiple users or processes hitting shared resources
- **Edge case inputs** — malformed data, empty values, unexpected types
- **Dependency interactions** — where one component's failure cascades into another

## Supported Languages

- Python (Flask, FastAPI, Streamlit, pandas, NumPy, SQLAlchemy, LangChain, and more)
- JavaScript/Node.js (React, Next.js, Express, Three.js, Prisma, Socket.io, and more)

36 dependency profiles included. Unrecognized dependencies get generic stress testing.

## Example Output

```
============================================================
  myCode Diagnostic Report
============================================================

We found some problems that could affect your Incident Tracker under real-world conditions.

- When your Incident Tracker is handling multiple users at once, requests timed out under load.
- When your Incident Tracker is downloading large responses, the system hits memory limits.
- When your Incident Tracker receives unexpected input, the code crashes instead of handling it gracefully.

Found 3 critical issue(s) and 25 warning(s) across 70 scenarios.
Performance degradation detected in 11 area(s).
```

## Offline Mode

Run without any API calls:

```bash
mycode ~/path/to/your/project --offline
```

Uses template-based scenario generation instead of LLM. Still runs all stress tests and produces a full report.

## Online Mode (Gemini Flash)

With an API key, myCode uses Gemini Flash to generate a natural-language summary of findings:

```bash
export GEMINI_API_KEY=your_key_here
mycode ~/path/to/your/project
```

Or pass it directly:

```bash
mycode ~/path/to/your/project --api-key your_key_here
```

Free tier: get a Gemini API key at https://aistudio.google.com/apikey (1,000 requests/day, no credit card).

## How It Works

1. **Ingester** parses your code (AST analysis), maps dependencies, identifies coupling points
2. **Conversational interface** asks what your project does and how you expect it to be used
3. **Scenario generator** creates stress test configurations based on your code + your intent
4. **Execution engine** runs your actual code in a sandbox with synthetic data under escalating conditions
5. **Report generator** translates raw results into plain-language diagnostics

Your original files are never modified. Everything runs in a temporary environment that is destroyed on completion.

## Safety

- Your code is copied into a temporary sandbox — originals are never touched
- Resource caps (memory, CPU, timeout) prevent runaway processes
- Sandbox is destroyed on completion, crash, or interrupt (Ctrl+C)
- No data is collected without explicit opt-in consent
- No source code is ever sent to any API — only project descriptions and dependency names

**Important:** myCode executes your project's code on your machine. Do not point it at code you do not trust.

## What myCode Is NOT

- Not a linter or code reviewer
- Not a security scanner
- Not a code generator — myCode never modifies your code
- Not a guarantee of code quality — it's a diagnostic tool

## Requirements

- Python 3.10+
- Node.js 18+ (for JavaScript project testing)

## Validated Against Real Projects

myCode has been tested against real-world projects across Python and JavaScript:

| Project | Language | Scenarios | Critical Issues | Result |
|---------|----------|-----------|-----------------|--------|
| Multi-framework web app (Flask + FastAPI + Streamlit) | Python | 70 | 3 | Timeouts under load, memory scaling issues |
| Analysis tool (React + charting libraries) | JavaScript | 25 | 4 | Edge case input crashes, memory growth |
| Multi-component framework (React) | JavaScript | 19 | 0 | All clean |
| Expense tracker (React) | JavaScript | 5 | 0 | Mostly clean |
| CLI tool — myCode self-test (26 files, 24K lines) | Python | 279 | 0 | All clean |

398 total scenarios, 7 critical issues found, 0 myCode errors.

[See full anonymized test reports →](docs/validation-reports.md)

## Contact

Machine.Adjacent.Systems@protonmail.com

## License

MIT

---

*myCode diagnoses — it does not prescribe. Interpret results in your context.*
