# myCode — Token Economics & Architectural Optimization
# Analysis conducted February 20, 2026
# CORRECTED with verified DeepSeek, Gemini, OpenAI, and Claude pricing
# Transfer this into the myCode project for continuity.

---

## CONTEXT

A previous session (now deleted due to platform failure) contained detailed cost modelling that arrived at a figure of $37 per 1,000 myCode sessions as the cost to MAS for the free tier. This document reconstructs the analysis, validates the $37 figure, identifies an architectural optimization, and provides comprehensive provider comparison for tier planning.

---

## VERIFIED API PRICING (February 2026)

### DeepSeek

| Model | Input/1M tokens | Output/1M tokens | Notes |
|-------|----------------|------------------|-------|
| Chat V3 | $0.27 (miss) / $0.07 (hit) | $1.10 | General tasks |
| Reasoner R1 | $0.55 (miss) / $0.14 (hit) | $2.19 | Complex reasoning |

- Free tier: 1M tokens/month, no credit card required
- Off-peak (16:30–00:30 UTC): 50% off Chat; 75% input / 50% output off Reasoner
- Cache hits: automatic on shared prefixes (system prompts, repeated context)
- Pro subscription: ~$0.50/month for priority queue

### Gemini

| Model | Input/1M tokens | Output/1M tokens | Notes |
|-------|----------------|------------------|-------|
| 1.5 Flash | $0.075 | $0.30 | Cheapest viable option |
| 1.5 Flash-Lite | $0.10 | $0.40 | High-volume |
| 2.5 Flash | $0.15 | $0.60 | Hybrid reasoning |
| 1.5 Pro | $1.25 | $5.00 | Doubles above 128K context |
| 2.5 Pro | $1.25 | $10.00 | Doubles above 200K context |
| 3 Pro (Preview) | $2.00 | $12.00 | Paid tier only |

- Free tier: up to 1,000 requests/day, no credit card required
- Context caching: ~$4.50/1M tokens/hour storage, reads at $0.15-0.30/M (75%+ savings)
- Batch processing: 50% discount for async workloads
- Free tier data may be used by Google; paid tier data is private

### OpenAI

| Model | Input/1M tokens | Output/1M tokens | Notes |
|-------|----------------|------------------|-------|
| GPT-5 | $1.25 | $10.00 | Advanced general |
| GPT-5 (cached) | $0.125 | $10.00 | 90% off input for repeated prompts |
| GPT-5 Mini | $0.25 | $2.00 | Mid-range |
| GPT-5 Mini (cached) | $0.025 | $2.00 | |
| GPT-5 Nano | $0.05 | $0.40 | Lowest cost |
| GPT-5 Nano (cached) | $0.005 | $0.40 | |

- No permanent free tier; introductory credits for new accounts
- Batch/Flex tiers: ~50% off for non-urgent workloads
- Prompt caching: automatic ~90% off input for identical repeated prompts

### Claude (Anthropic)

| Model | Input/1M tokens | Output/1M tokens |
|-------|----------------|------------------|
| Sonnet 4.6 | $3.00 | $15.00 |
| Opus 4.6 | $15.00 | $75.00 |

- No free tier for API
- Payment issue currently unresolved (card accepted for Max subscription but rejected on Console)

---

## TOKEN CONSUMPTION MODEL

### Test scenario:
- Real-world vibe-coded project: FastAPI + SQLAlchemy + Pandas CSV analyzer with Streamlit frontend
- 12 Python files, ~1,800 lines, 7 dependencies
- User intent: small business CSV upload tool, 10 users, files up to 50MB
- Full prompt construction details in companion file: myCode-Token-Simulation.md

### Three LLM calls per myCode session:
1. **Conversational Interface** — extracts user's operational intent in plain language
2. **Scenario Generator** — produces executable stress test configurations
3. **Report Generator** — translates raw execution results into plain-language diagnostic

---

## UNOPTIMIZED ARCHITECTURE (ORIGINAL)

Full ingester output + all component library profiles sent to conversational interface on every turn. 4 conversational turns, each resending full context.

| Call | Input Tokens | Output Tokens |
|------|-------------|---------------|
| 1. Conversational Interface (4 turns, full context) | 66,600 | 1,800 |
| 2. Scenario Generator | 17,100 | 4,000 |
| 3. Report Generator | 13,300 | 5,300 |
| **TOTAL** | **97,000** | **11,100** |

Call 1 consumes 69% of all input tokens.

### Unoptimized cost per 1,000 sessions:

| Provider / Model | Cost/1,000 | Cost/session |
|-----------------|-----------|-------------|
| **DeepSeek V3 (no caching)** | **$38.40** | **$0.0384** |
| DeepSeek V3 (caching, ~60% hit) | $26.50 | $0.0265 |
| DeepSeek V3 (caching + off-peak) | $13.25 | $0.0133 |
| Gemini 1.5 Flash | $10.61 | $0.0106 |
| Gemini 2.5 Flash | $21.21 | $0.0212 |
| Gemini 1.5 Pro | $176.75 | $0.1768 |
| Gemini 2.5 Pro | $232.25 | $0.2323 |
| GPT-5 Nano | $9.14 | $0.0091 |
| GPT-5 Nano (cached) | $4.99 | $0.0050 |
| GPT-5 Mini | $46.45 | $0.0465 |
| GPT-5 Mini (cached) | $24.63 | $0.0246 |
| GPT-5 | $232.25 | $0.2323 |
| GPT-5 (cached) | $123.22 | $0.1232 |
| Claude Sonnet 4.6 | $457.50 | $0.4575 |
| Claude Opus 4.6 | $2,287.50 | $2.2875 |

### THE $37 FIGURE — VALIDATED

**DeepSeek V3 (no caching) at 97,000 input / 11,100 output = $38.40 per 1,000 sessions.** This matches the $37 from the deleted session, confirming it was DeepSeek V3 against the unoptimized architecture with no caching or off-peak discount applied.

---

## ARCHITECTURAL OPTIMIZATION

Three changes that cut token costs 40-50%:

### 1. Remove component library profiles from Call 1
Conversational interface doesn't need library profiles — it asks users about their project in plain language. Profiles go to Scenario Generator (Call 2) only. Saves ~8,000 tokens per turn × 4 turns = ~32,000 tokens.

### 2. Summarize ingester output for Call 1
Conversational interface gets a ~500-token summary instead of the full AST dump, dependency tree, and function flow map. Full output goes to Calls 2 and 3 only.

### 3. Reduce conversational turns from 4 to 2
Ingester already provides most project information. Conversation confirms user intent and operational context in 2 turns, not 4.

**Result: Call 1 drops from 66,600 input tokens to ~8,000.**

---

## OPTIMIZED ARCHITECTURE

| Call | Input Tokens | Output Tokens |
|------|-------------|---------------|
| 1. Conversational Interface (2 turns, lean context) | 8,000 | 1,000 |
| 2. Scenario Generator (full context) | 17,100 | 4,000 |
| 3. Report Generator (full results) | 13,300 | 5,300 |
| **TOTAL** | **38,400** | **10,300** |

### Optimized cost per 1,000 sessions:

| Provider / Model | Before | After | Savings |
|-----------------|--------|-------|---------|
| **DeepSeek V3 (no caching)** | **$38.40** | **$21.71** | **43%** |
| DeepSeek V3 (caching, ~60% hit) | $26.50 | $16.30 | 38% |
| DeepSeek V3 (caching + off-peak) | $13.25 | $8.15 | 38% |
| Gemini 1.5 Flash | $10.61 | $6.18 | 42% |
| Gemini 2.5 Flash | $21.21 | $11.94 | 44% |
| Gemini 1.5 Pro | $176.75 | $103.50 | 41% |
| Gemini 2.5 Pro | $232.25 | $138.00 | 41% |
| GPT-5 Nano | $9.14 | $5.00 | 45% |
| GPT-5 Nano (cached) | $4.99 | $3.16 | 37% |
| GPT-5 Mini | $46.45 | $25.38 | 45% |
| GPT-5 Mini (cached) | $24.63 | $14.67 | 40% |
| GPT-5 | $232.25 | $138.00 | 41% |
| GPT-5 (cached) | $123.22 | $75.19 | 39% |
| Claude Sonnet 4.6 | $457.50 | $270.00 | 41% |
| Claude Opus 4.6 | $2,287.50 | $1,350.00 | 41% |

---

## ADDITIONAL COST REDUCTION STRATEGIES

### DeepSeek:
- **Cache hits (90% off input):** Automatic on shared prefixes. System prompt + ingester summary cached after first turn.
- **Off-peak (50% off):** 16:30–00:30 UTC. Not for real-time sessions but relevant for batch features.
- **Model choice:** V3 for all calls unless scenario generation quality demands R1 (2x cost).

### Gemini:
- **Context caching (75%+ off reads):** Cache system prompts and library profiles. $4.50/M tokens/hour storage.
- **Batch processing (50% off):** For non-real-time workloads.
- **Model tiering:** 1.5 Flash for conversation ($0.075/M), 2.5 Pro for scenario generation ($1.25/M).
- **Free tier:** 1,000 requests/day. During beta/launch, may cover all MAS needs at zero cost.

### OpenAI:
- **Prompt caching (90% off input):** Automatic for identical repeated prompts. Very effective for system prompts.
- **Batch/Flex (50% off):** For non-urgent processing.
- **Model tiering:** Nano for conversation + report, Mini or GPT-5 for scenario generation.

### General:
- **Right-size per call:** Cheapest model for conversation (simple extraction), best model for scenario generation (complex reasoning), mid-tier for report (summarization).
- **Prompt engineering:** Every 1,000 tokens cut from system prompt saves across every session.
- **Limit output tokens:** Set max_tokens on each call to prevent runaway generation.

---

## PROVIDER COMPARISON: BEST OPTIONS PER TIER

### Free tier — cheapest options (optimized, per 1,000 sessions):

| Option | Cost/1,000 | Notes |
|--------|-----------|-------|
| GPT-5 Nano (cached) | $3.16 | Cheapest. Quality unknown for scenario generation. No free tier. |
| Gemini 1.5 Flash | $6.18 | Very cheap. 1,000 free requests/day. Quality adequate? |
| DeepSeek V3 (caching + off-peak) | $8.15 | Cheap but off-peak limits real-time use. |
| Gemini 2.5 Flash | $11.94 | Good balance. Free tier available. |
| DeepSeek V3 (caching) | $16.30 | Proven quality. Known pricing. |
| DeepSeek V3 (no caching) | $21.71 | Conservative baseline. |

### Recommended blends for free tier (optimized):

| Blend | Conv (Call 1) | Scenarios (Call 2) | Report (Call 3) | Est. cost/1,000 |
|-------|--------------|-------------------|----------------|-----------------|
| DeepSeek only | V3 | V3 | V3 | $16-22 |
| Gemini tiered | 1.5 Flash | 2.5 Pro | 1.5 Flash | ~$18 |
| GPT tiered | Nano | Mini | Nano | ~$12 |
| Cross-provider | Gemini 1.5 Flash | DeepSeek V3 | Gemini 1.5 Flash | ~$10 |

**Key trade-off:** Cheapest models (Nano, 1.5 Flash) may not generate quality stress scenarios. Scenario generation requires reasoning about code behavior — must test quality before committing to cheap blend.

### Freemium tier (optimized, per 1,000 sessions):

| Option | Cost/1,000 | Notes |
|--------|-----------|-------|
| GPT-5 (cached) | $75.19 | Best value if caching works well |
| Gemini 2.5 Pro | $138.00 | Strong reasoning |
| GPT-5 | $138.00 | Identical pricing to Gemini Pro |
| Claude Sonnet 4.6 | $270.00 | Most expensive. Premium positioning only. |

### Enterprise tier (optimized, per 1,000 sessions):

| Option | Cost/1,000 | Notes |
|--------|-----------|-------|
| Claude Sonnet 4.6 | $270.00 | Strong reasoning, premium brand |
| Gemini 3 Pro | ~$350 (est.) | Preview pricing, may decrease |
| Claude Opus 4.6 | $1,350.00 | Best quality, highest cost |

---

## PRICING IMPLICATIONS

### Free tier (DeepSeek V3 / Gemini Flash / GPT-5 Nano blend):

**Optimized cost range: $8-22 per 1,000 runs depending on caching and model blend.**

| Monthly free runs | Cost to MAS (low) | Cost to MAS (high) |
|-------------------|-------------------|-------------------|
| 10,000 | $80 | $220 |
| 50,000 | $400 | $1,100 |
| 100,000 | $800 | $2,200 |

Manageable at early volumes. Rate limiting via hashed machine ID controls abuse. DeepSeek and Gemini free tier quotas may cover initial launch period at near-zero cost.

### Freemium tier:

**Using Gemini 2.5 Pro or GPT-5 (optimized): ~$138/1,000 runs = $0.138/run.**

| Price point | Included runs | MAS cost | Margin |
|-------------|--------------|----------|--------|
| $15/month | 50 | $6.90 | 54% |
| $20/month | 75 | $10.35 | 48% |
| $20/month | 100 | $13.80 | 31% |
| $30/month | 100 | $13.80 | 54% |
| $30/month | 150 | $20.70 | 31% |

**Using Claude Sonnet (optimized): ~$270/1,000 runs = $0.27/run.**

| Price point | Included runs | MAS cost | Margin |
|-------------|--------------|----------|--------|
| $20/month | 50 | $13.50 | 32% |
| $20/month | 100 | $27.00 | **-35% (LOSS)** |
| $30/month | 75 | $20.25 | 32% |
| $40/month | 100 | $27.00 | 32% |

**Gemini Pro or GPT-5 is significantly more viable than Claude Sonnet for freemium.** Claude at $20/month with 100 runs is loss-making.

### Enterprise tier:

| Provider | Cost/run | Minimum viable annual contract |
|----------|----------|-------------------------------|
| Claude Sonnet | $0.27 | $500-2,000/year with usage caps |
| Claude Opus | $1.35 | $2,000-5,000/year with usage caps |
| Gemini 3 Pro | ~$0.35 | $1,000-3,000/year |

Enterprise pricing must be custom contracts.

---

## RECOMMENDED PROVIDER STRATEGY

| Tier | Primary | Backup | Cost/1,000 (optimized) |
|------|---------|--------|----------------------|
| Free | DeepSeek V3 | Gemini 2.5 Flash | $12-22 |
| Freemium | Gemini 2.5 Pro or GPT-5 | Claude Sonnet | $75-138 |
| Enterprise | Claude Sonnet | Claude Opus (premium) | $270-1,350 |

BYOK (Bring Your Own Key) option allows users to use their own API keys for any supported provider, bypassing MAS proxy costs entirely.

---

## REQUIRED CHANGE TO CLAUDE.md

The Conversational Interface (Component 4) specification must be updated:

1. Receives a **summarized ingester output** (~500 tokens), not the full AST/dependency/flow analysis
2. Receives **no component library profiles** — those go to Scenario Generator only
3. Target is **2 conversational turns**, not 3-5
4. Full ingester output and component library profiles passed to Calls 2 and 3 only

Architectural change driven by token economics. Does not change user experience.

---

## OUTSTANDING ITEMS

1. **API payment issue.** Card works for Claude Max but rejected on Anthropic Console. Bank confirms no block. Contact Anthropic support. Try DeepSeek and Gemini sign-ups separately — different payment processors may accept same card.

2. **Real simulation needed.** Theoretical estimates validated the $37 figure but actual projects may vary. When API access is available, run real prompts and confirm.

3. **Free tier quotas at launch:**
   - DeepSeek: 1M tokens/month free. One optimized session ≈ 48,700 tokens. Free tier covers ~20 sessions/month. Insufficient for user-facing free tier — must be on paid API.
   - Gemini: 1,000 requests/day free. One session = 3 API calls (or 4-6 with multi-turn). Free tier covers ~167-333 sessions/day. Potentially sufficient for early launch.
   - OpenAI: Introductory credits only. Must be on paid API.

4. **Quality validation.** Cheapest models (GPT-5 Nano, Gemini 1.5 Flash) are untested for scenario generation quality. Scenario generation requires reasoning about code behavior under stress — may need a more capable model. Test before committing to a cheap blend.

5. **Freemium backend decision.** Gemini 2.5 Pro and GPT-5 are identically priced ($1.25/$10) and both viable. Decision should be based on scenario generation quality, not cost. Test both.

6. **Context caching implementation.** DeepSeek auto-caches on shared prefixes. Gemini requires explicit caching setup with storage costs. OpenAI auto-caches identical prompts. Architecture should maximize cache hits — fixed system prompts, stable ingester summary format, consistent library profile structure.
