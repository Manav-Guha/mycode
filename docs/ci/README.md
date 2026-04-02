# myCode CI Gate

Run myCode automatically on every pull request. Get stress-test results as a GitHub status check.

## Quick Start

### 1. Get an API key

Contact Machine.Adjacent.Systems@protonmail.com to request a CI API key. Keys are prefixed `mck_` and shown only once — save it securely.

### 2. Add secrets to your repo

Go to your GitHub repo → Settings → Secrets and variables → Actions.

**Repository secret:**
- Name: `MYCODE_API_KEY`
- Value: your `mck_` key

**Repository variable:**
- Name: `MYCODE_API_URL`
- Value: `https://mycode-production-d3fa.up.railway.app`

### 3. Add the workflow

Copy `mycode-check.yml` from this directory into your repo at `.github/workflows/mycode-check.yml`.

Or create the file manually — see the template below.

### 4. Open a pull request

The myCode check runs automatically on every PR to `main`. Results appear as a status check on the PR.

## Configuration

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `MYCODE_API_KEY` | Yes | — | Your CI API key (secret) |
| `MYCODE_API_URL` | Yes | — | myCode backend URL (variable) |
| `MYCODE_THRESHOLD` | No | `report_only` | When to fail: `report_only`, `critical`, `warning`, `any` |
| `MYCODE_TIER` | No | `2` | Analysis depth: `1` (fast inference), `2` (standard), `3` (deep) |

## Thresholds

- **`report_only`** (default) — never blocks the PR. Shows results as an informational check. Start here.
- **`critical`** — blocks the PR if any critical finding is detected.
- **`warning`** — blocks on critical or warning findings.
- **`any`** — blocks on any finding at all.

## Overriding a Failed Check

If myCode blocks a PR and you believe it's a false positive:

```bash
curl -X POST "https://your-mycode-url/api/ci/override/{job_id}" \
  -H "Content-Type: application/json" \
  -d '{"api_key": "mck_your_key", "reason": "false positive on dep check"}'
```

The override is logged with your key and reason for audit purposes. The PR check flips to pass.

## How It Works

1. PR is opened → GitHub Actions triggers the workflow
2. Workflow calls `POST /api/ci/check` with your repo URL and settings
3. myCode clones the repo, runs analysis (same pipeline as the web interface)
4. Workflow polls `GET /api/ci/result/{job_id}` every 15 seconds
5. When complete, the check passes or fails based on your threshold
6. Full report available at the report URL in the check output

Analysis typically takes 3-7 minutes depending on project size and tier.

## Notes

- myCode infrastructure errors (timeouts, server issues) never block your PR — the check exits with a warning, not a failure.
- The CI gate uses the same analysis engine as the web interface. Same corpus, same predictions, same reports.
- Each CI run is logged with `source=ci` for analytics.

## Template

See [mycode-check.yml](mycode-check.yml) in this directory for the complete GitHub Actions workflow template.
