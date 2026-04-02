# myCode

Stress-test your AI-generated code before it breaks.

You built it with AI. myCode finds where it breaks — and at what point.

## Try It

**Web interface:** [mycode-ai.vercel.app](https://mycode-ai.vercel.app/) — paste a GitHub URL, describe your project, get a report. No account required.

**CLI:**
```bash
pip install mycode-ai
mycode /path/to/your/project
```

**CI gate:** Add myCode to your GitHub Actions pipeline. Every PR gets stress-tested automatically. [See docs/ci/](docs/ci/).

## What It Does

myCode takes a GitHub repo that you provide (via URL or zip upload), installs dependencies in an isolated sandbox, and runs runtime stress tests against your stated intent — "I want this to handle 500 users, transacting 10GB of data, in burst traffic."

It doesn't review your code. It runs your code and tells you where and why it breaks.

**Before any tests run**, a prediction model trained on 6,000+ repos predicts what's likely to fail based on your project's dependency profile and architecture. The tests then confirm or refute those predictions.

The report includes fix prompts you can paste directly into your coding agent (Claude Code, Cursor, Copilot, etc.).

## How It Works

1. Submit a GitHub repo URL or upload a zip file.
2. Describe your project: what it does, expected users, data volumes, usage patterns.
3. myCode clones the repo, detects the language and framework, installs dependencies in an isolated venv, and classifies the project architecture (API service, dashboard, data pipeline, etc.).
4. An XGBoost prediction model (56 features, 40 target patterns, 0.91 mean AUC) predicts likely failure modes.
5. myCode runs dependency-specific stress tests, edge case validation, and HTTP load testing for server frameworks.
6. You get a report: what broke, at what load level, degradation curves showing where performance falls off a cliff, and actionable fix prompts.

All tests run inside isolated temporary environments. Your original files are never touched.

## Self-Test

myCode was built with Claude Code. We ran it against its own codebase. It found that the `/api/preflight` endpoint degrades from under 1ms to over 15 seconds at 50 concurrent connections, correctly diagnosing thread pool saturation. [See the full self-test report](docs/self-test/).

## Supported Languages

**Python (primary):** FastAPI, Flask, Django, Streamlit, Gradio, pandas, NumPy, SQLAlchemy, Supabase, LangChain, LlamaIndex, ChromaDB, OpenAI SDK, Anthropic SDK, requests, httpx, Pydantic, and more — 30+ dependency profiles.

**JavaScript/TypeScript:** Dependency detection and basic analysis in place. Full runtime testing on the roadmap.

Dependencies not in the library are flagged as unrecognised and tested generically. If your project uses an unsupported language (Kotlin, Java, Go, etc.), myCode will let you know.

## Stress Categories

- **Data volume scaling** — progressively larger inputs until something breaks
- **Memory profiling** — tracking accumulation over repeated runs
- **Edge case inputs** — malformed, empty, unexpected type data
- **Concurrent execution** — multiple instances against shared resources
- **HTTP load testing** — endpoint degradation under concurrent connections
- **Computation coupling** — processing time growth with input size

## What a Report Contains

- Project summary with architecture classification
- Predictive analysis from the corpus (before tests run)
- Findings grouped by severity: priority improvements, opportunities, good to know
- Per-finding: what broke, at what load level, degradation curves, and a fix prompt for your coding agent
- Scaling roadmap: low load → mid load → peak load with verdict per metric
- Dependency profile showing corpus-wide failure patterns for your stack
- PDF and JSON downloads

## CI Gate

myCode can run as a CI check on every pull request. The default mode is non-blocking (report only). You can configure it to block merges on critical findings.

```yaml
# .github/workflows/mycode.yml
name: myCode Check
on: [pull_request]
jobs:
  stress-test:
    runs-on: ubuntu-latest
    steps:
      - name: Run myCode
        env:
          MYCODE_API_KEY: ${{ secrets.MYCODE_API_KEY }}
          MYCODE_API_URL: ${{ vars.MYCODE_API_URL }}
          MYCODE_THRESHOLD: report_only  # or: critical, warning, any
        run: |
          # See docs/ci/mycode-check.yml for full template
```

[Full CI setup instructions →](docs/ci/)

## The Corpus

myCode's predictions are backed by a corpus of 6,000+ empirically tested repositories. The prediction model retrains as the corpus grows. Architecture-specific predictions (e.g., "63% of API service projects experience server startup failures") come from real test data, not heuristics.

## Numbers

- **6,000+** repos in the corpus
- **0.91** mean AUC across 40 prediction targets (5-fold cross-validation)
- **2,600+** tests in the codebase
- **30+** dependency profiles with stress scenarios

## What myCode Is Not

- Not a replacement for your coding agent — it complements Cursor, Claude Code, Copilot, and similar
- Not a security scanner
- Not a code generator — myCode never writes or modifies code
- Not a guarantee — diagnostic tool that shows you where problems are

## Business Model

myCode is free right now. We are pre-revenue and figuring out where the paywall belongs. The likely split: free tier gives the full diagnostic, paid tier adds predictive intelligence and continuous monitoring.

## License

Source-available under the **Business Source License 1.1** (BSL 1.1). The source code is public for transparency. Production use requires a commercial license. Each version converts to **Apache License 2.0** four years after its first public distribution — the same model used by HashiCorp, Sentry, and CockroachDB.

See [LICENSE](LICENSE) for the full text.

**Licensor:** Manabrata Guha
**Licensed Work:** myCode
**Change License:** Apache License 2.0
**Change Date:** Four years from first public distribution of each version

## Contact

Machine.Adjacent.Systems@protonmail.com
