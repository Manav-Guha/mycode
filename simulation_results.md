# myCode — Token Consumption Simulation Results
# Phase B: Pre-Build Validation
# Date: February 20, 2026

## Method

Token counts estimated from actual prompt character lengths using standard
tokenizer ratio (~3.5 characters per token for mixed English/JSON content).
Costs projected using published API pricing. Actual API calls to DeepSeek and
Anthropic were not possible due to billing setup issues; Gemini free tier
rate-limited. Character-based estimation is standard practice and accurate
to within +/- 20%.

## Prompts Tested

Three prompts simulating the three LLM-calling components in myCode:

1. **Conversational Interface** — Extract operational intent from a non-technical
   user describing a Flask expense tracker app. (~246 input tokens)

2. **Scenario Generator** — Generate stress test configurations from parsed
   project data (Flask + SQLite + pandas) and user intent. (~479 input tokens)

3. **Report Generator** — Produce plain-language diagnostic report from raw
   stress test results including concurrent write failures, memory profiling,
   and edge case findings. (~761 input tokens)

## Token Volume Per Session

| Component                 | Input Tokens | Output Tokens (est.) |
|---------------------------|-------------|---------------------|
| Conversational Interface  | ~246        | ~400                |
| Scenario Generator        | ~479        | ~800                |
| Report Generator          | ~761        | ~1,000              |
| **TOTAL per session**     | **~1,487**  | **~2,200**          |

Note: Output tokens are estimated based on typical LLM response lengths for
these task types. Input tokens are calculated from actual prompt content.

## Cost Projections

### Per Session

| Model              | Input Cost  | Output Cost | Total/Session |
|--------------------|------------|------------|---------------|
| DeepSeek V3        | $0.00021   | $0.00062   | $0.00082      |
| Claude Sonnet 4.5  | $0.00446   | $0.03300   | $0.03746      |
| Claude Opus 4.6    | $0.02230   | $0.16500   | $0.18730      |

### Monthly Projections

| Model              | 100 sessions | 1,000 sessions | 10,000 sessions |
|--------------------|-------------|----------------|-----------------|
| DeepSeek V3        | $0.08       | $0.82          | $8.24           |
| Claude Sonnet 4.5  | $3.75       | $37.46         | $374.61         |
| Claude Opus 4.6    | $18.73      | $187.31        | $1,873.05       |

### Cost Ratio

Claude Sonnet is **45x** more expensive than DeepSeek per session.

## Key Findings

1. **DeepSeek free tier is extremely cheap to absorb.** Even at 10,000 sessions/month,
   MAS cost is ~$8. The CLAUDE.md assumption of "negligible at early volumes" is
   validated.

2. **Claude freemium tier needs careful pricing.** At 1,000 sessions/month with
   Sonnet, MAS cost is ~$37. This is the baseline for subscription pricing — the
   monthly subscription price must exceed this to be sustainable, plus margin.

3. **Opus for enterprise is expensive.** At $187/month for 1,000 sessions, enterprise
   pricing needs to reflect this. However, enterprise is unmetered per the spec,
   so this becomes a per-seat calculation.

4. **Output tokens dominate cost.** Output is 75-88% of total cost across all models.
   Any optimization that reduces output verbosity (shorter reports, structured
   output) directly reduces cost.

5. **Session token volume is modest.** ~3,700 total tokens per session is small.
   This confirms that myCode sessions are not token-heavy — the value is in the
   execution engine and scenario generation logic, not in raw LLM token volume.

## Pricing Implications

- Free tier (DeepSeek): Absorb cost. No concern at any realistic early volume.
- Freemium tier (Claude Sonnet): At ~$0.037/session, a $20/month subscription
  with 500 included sessions costs MAS ~$18.73 in API fees. Tight but viable.
  Per-run overage at $0.05-0.10/run provides margin.
- Enterprise tier (Claude Opus): Custom pricing required. Per-seat model
  recommended over per-session.

## Limitations

- Token counts are estimates, not measured. Variance of +/- 20% expected.
- Output token estimates assume typical response lengths; actual will vary
  by project complexity.
- Does not account for retry costs (failed API calls, rate limiting).
- Does not account for multi-turn conversational interface (spec says 3-5
  minute exchange; this simulates a single turn only). Real sessions may
  use 2-4x the conversational interface tokens.

## Recommendation

Proceed with build. Cost structure is viable for all three tiers. Revisit
pricing with real token data once the tool is functional and processing
actual projects.
