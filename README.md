# myCode

Runtime stress-testing and reliability verification for AI-generated applications.

You built it with AI. myCode finds where it breaks.

## What It Does

myCode takes a GitHub repository, runs the actual code under progressively escalating conditions, and produces a diagnostic report showing where and how it fails. It tests dependency interactions as systems, not individual components in isolation.

It is not a linter, not a static analysis tool, not a code reviewer, and not a security scanner. It does not generate patches or modify your code. It runs your code, breaks it under controlled conditions, and tells you what happened.

## How It Works

1. Point myCode at a GitHub repo URL via the [web interface](https://mycode-ai.vercel.app/).
2. Answer 2–3 questions about your project in plain language (what it does, who uses it, what conditions it operates under).
3. myCode clones the repo, resolves dependencies, parses the codebase via AST, matches against its component library of 36 dependency profiles, generates stress scenarios from the intersection of your stated intent and parsed code structure, then executes them.
4. You get a diagnostic report: what broke, at what load level, and under what conditions.

All tests run inside isolated temporary environments. Your original files are never touched.

## Supported Languages and Frameworks

**Python (18 profiles):** Flask, FastAPI, Streamlit, Gradio, pandas, NumPy, SQLite3, SQLAlchemy, Supabase, LangChain, LlamaIndex, ChromaDB, OpenAI SDK, Anthropic SDK, requests, httpx, Pydantic, os/pathlib.

**JavaScript / Node.js (18 profiles):** React, Next.js, Express, Node.js core (fs, path, http), Tailwind CSS, Three.js, Svelte, OpenAI Node SDK, Anthropic Node SDK, LangChain.js, Supabase JS, Prisma, Axios/fetch, Mongoose/MongoDB driver, Stripe SDK, dotenv, Zod, Socket.io.

Dependencies not in the library are flagged as unrecognised and tested generically based on how the code uses them.

## Stress Categories

- **Data volume scaling** — progressively larger inputs until something gives
- **Memory profiling** — repeated runs tracking accumulation over time
- **Edge case inputs** — malformed, empty, unexpected type data
- **Concurrent execution** — multiple instances against shared resources
- **Language-specific** — GIL contention and blocking I/O (Python); async/promise chain failures, event listener accumulation, state management degradation (JavaScript)

## Installation

### Web Interface (recommended)

Go to [mycode-ai.vercel.app](https://mycode-ai.vercel.app/). Paste a GitHub repo URL. No account required.

### CLI

```bash
pip install mycode-ai
```

```bash
mycode
```

The CLI walks you through the same conversational flow. Requires Python 3.10+.

## Offline and Online Modes

**Offline mode** (default, no API key): Scenarios are generated from template-based matching against the component library. Reports use structured templates. No external API calls.

**Online mode** (Gemini Flash API key): Adds LLM-generated natural language scenario descriptions and report summaries. Set your API key when prompted or via environment variable.

## Test Suite

1,694 passing tests across all components.

## What a Report Contains

- Project summary and detected dependency stack
- Findings grouped by severity (critical, warning, info)
- Per-finding: what broke, at what scale step, under what conditions
- Unrecognised dependencies flagged with generic test results
- Probe-and-skip classification for functions requiring runtime context (e.g., Streamlit session state, database connections) — reported honestly as untestable in isolation, not as false failures

Reports diagnose. They do not prescribe fixes.

## What myCode Is Not

- Not a replacement for your coding tool — it complements Cursor, Claude Code, Copilot, Lovable, and similar.
- Not a guarantee of code quality — diagnostic tool only.
- Not a security scanner.
- Not a code generator — myCode never writes or modifies code.

## License

myCode is source-available under the **Business Source License 1.1** (BSL 1.1).

### What this means

The complete source code is publicly available. You can read it, study it, fork it, and build on it for non-production purposes. **Production use of myCode requires a commercial license from the licensor.**

### What you can do without a commercial license

- **Evaluate and test** myCode in development and non-production environments.
- **Read, study, and learn** from the source code.
- **Fork and modify** the code for non-production use.
- **Contribute** to the project via pull requests.

### What requires a commercial license

- **Any production use** — running myCode against your projects in a production or commercial context.
- **Offering myCode as a service** — hosting myCode or a derivative and providing it to third parties.

### Why BSL

The source is open for transparency and scrutiny. The license protects the project commercially while the product is young. This is the same license used by HashiCorp, Sentry, CockroachDB, and SurrealDB.

### Conversion to open source

Each version of myCode converts to the **Apache License 2.0** four years after its first public distribution. At that point, all restrictions lapse and the code is fully open source under Apache 2.0.

**Licensor:** Manabrata Guha
**Licensed Work:** myCode
**Additional Use Grant:** None
**Change License:** Apache License 2.0
**Change Date:** Four years from first public distribution of each version

The full license text is in [LICENSE](LICENSE).

For commercial licensing enquiries: Machine.Adjacent.Systems@protonmail.com

## Contact

Machine.Adjacent.Systems@protonmail.com
