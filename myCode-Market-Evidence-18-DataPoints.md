# myCode Market Evidence — 18 Data Points
## Machine Adjacent Systems (MAS) — Updated February 28, 2026

---

## Summary

18 data points collected between February 22–28, 2026, documenting the validation gap in AI-generated code. Sources include Financial Times, Anthropic research, Columbia University, VentureBeat, The New Stack, InfoWorld, security researchers, and a US Department of Defense solicitation.

**Core thesis confirmed:** AI code generation has outpaced quality assurance. The bottleneck is no longer writing code — it is verifying that AI-generated code actually works as intended under real-world conditions.

---

## Data Points 1–10 (Sessions 2–5)

| # | Source | Date | Key Finding | Investor Framing |
|---|--------|------|-------------|------------------|
| 1 | FT — Oliver Roeder | Feb 23, 2026 | Built word processor in a weekend with AI, zero testing mentioned | "People are shipping AI-built software with no verification step" |
| 2 | Boris Cherny (Claude Code creator) | Feb 22–23, 2026 | Job title "software engineer" may become obsolete; "failure modes may become foundational skill" | The creator of AI coding tools says understanding failure is the new core skill |
| 3 | Anthropic — Skill Formation Study | Jan 29, 2026 | AI users score 17% lower on independent debugging tasks | AI makes people faster but worse at catching problems |
| 4 | Anthropic — COBOL/IBM story | Feb 23, 2026 | IBM fell 13% on AI modernisation fears; legacy code migration needs verification | Enterprise code migration at scale creates massive verification demand |
| 5 | Anthropic — Agent Autonomy Study | Feb 22, 2026 | 91% of AI code generation steps receive no human review | Only 9% of AI-generated code gets reviewed before shipping |
| 6 | InfoWorld — Matt Asay | Feb 23, 2026 | "True bottleneck is validation, not code generation" | Direct market thesis validation from major tech publication |
| 7 | Dario Amodei interview | Feb 2026 | AI coding acceleration transforming software development | CEO of leading AI company confirms code generation speed outpacing quality |
| 8 | WSJ enforcement article | Feb 2026 | Regulatory implications of AI-generated code | Regulatory environment creating compliance demand for verification |
| 9 | JPost vibe coding article | Feb 2026 | Mainstream awareness of vibe coding phenomenon | Market awareness expanding beyond tech circles |
| 10 | NDTV firing incident | Feb 2026 | Employee terminated for submitting AI-generated code that failed | Real-world career consequences of unverified AI code |

---

## Data Points 11–13 (Session 6–7: Security Evidence)

### Data Point 11: Claude Code CVEs (Check Point Research, Feb 25–26, 2026)
- **CVE-2025-59536 (CVSS 8.7):** Remote code execution via Hooks and MCP consent bypass. Malicious `.claude/settings.json` or `.mcp.json` executes arbitrary shell commands when project is opened in Claude Code. Fixed in v1.0.87 and v1.0.111.
- **CVE-2026-21852 (CVSS 5.3):** API key exfiltration. `ANTHROPIC_BASE_URL` in project config redirects all API traffic to attacker-controlled server. Fixed in v2.0.65.
- Attack vector: Supply chain. A single malicious commit compromises every developer who opens the repo.
- **Investor framing:** "AI coding tools themselves are attack surfaces. Configuration files that were passive data are now execution paths."
- Source: https://research.checkpoint.com/2026/rce-and-api-token-exfiltration-through-claude-code-project-files-cve-2025-59536/

### Data Point 12: Lovable EdTech Breach (VolodsTaimi, Feb 26, 2026)
- Lovable showcased EdTech app as a success story. 100K+ views. Real users from UC Berkeley, UC Davis, schools across Europe, Africa, Asia.
- Security researcher found 16 vulnerabilities in hours. 6 critical.
- **Auth logic inverted:** blocked logged-in users, let anonymous ones through. Behavioural bug — linters and static analysis tools would not catch it.
- 18,697 user records exposed (names, emails, roles) with no authentication required.
- Account deletion, grade modification, bulk email — all unauthenticated.
- Enterprise data from 14 institutions compromised.
- Lovable closed the support ticket without fixing the vulnerabilities.
- **Investor framing:** "$6.6B platform showcased app as success story. It had backwards auth logic exposing 18,697 users. Platform closed the bug report."
- Source: LinkedIn post by VolodsTaimi

### Data Point 13: Escape.tech — 2,000+ Vulnerabilities at Scale (Oct 2025)
- 5,600 publicly available vibe-coded apps scanned across Lovable (~4,000), Base44 (~159), Create.xyz (~449), Vibe Studio, Bolt.new
- 2,000+ vulnerabilities, 400+ exposed secrets, 175 PII instances (medical records, IBANs, phone numbers)
- Lovable-Supabase integration: anonymous JWT tokens in JS bundles, PostgREST APIs with permissive defaults
- CVE-2025-48757: Row Level Security bypass via public API keys
- **Investor framing:** "Researchers scanned 5,600 vibe-coded apps. Found 2,000+ vulnerabilities, 400+ exposed secrets, 175 PII leaks. This is the lower bound."
- Source: https://escape.tech/blog/methodology-how-we-discovered-vulnerabilities-apps-built-with-vibe-coding/

---

## Data Points 14–15 (Session 7: Market Maturation Evidence)

### Data Point 14: "From Vibes to Engineering" (The New Stack, Feb 26, 2026)
- Andrej Karpathy wants to retire "vibe coding" in favour of "agentic engineering" — the practice matured faster than the terminology.
- Forrester predicted this shift in Q4 2024: "Vibe coding will transform into vibe engineering by the end of 2026."
- Caylent CTO: "The differentiator isn't which LLM you picked, it's the agentic harness — tools, context management, evaluation, observability infrastructure."
- Cognitive debt (accumulated cost of poorly managed AI interactions, context loss, unreliable agent behavior) is 2026's primary threat.
- Resolve AI CEO: "The bottleneck isn't generating code anymore. It's understanding what's happening when that code breaks."
- **Investor framing:** Industry leaders now explicitly say the bottleneck is verification, not generation. myCode addresses the stated gap.

### Data Point 15: Vibe Coding Geography (LeadsNavi via The New Stack, Feb 5, 2026)
- Google search data for "vibe coding" analysed across 24 countries.
- Top 5 by search volume per 100K: Switzerland (41.19), Germany (40.29), Canada (37.78), Sweden (35.04), Finland (34.69).
- Europe dominates top 10. US ranks 15th (26.78/100K).
- Study excluded Asian countries (Google not primary search engine) and Middle East — UAE/Abu Dhabi market completely unmapped.
- Constellation Research analyst: US leading in actual enterprise usage despite lower search volume.
- Common associated searches: "vibe coding tools", "Claude code", "lovable", "bolt"
- **Investor framing:** Global demand, European early adoption, Middle East is white space. Hub71 positioning as first mover in unmapped region.

---

## Data Points 16–18 (Session 8: New Evidence)

### Data Point 16: Anthropic Claude Code Security — 500+ Zero-Days (VentureBeat, Feb 24, 2026)
- Anthropic pointed Claude Opus 4.6 at production open-source codebases and found 500+ high-severity vulnerabilities that survived decades of expert review and millions of hours of fuzzing.
- Launched Claude Code Security on Feb 20 as limited research preview for Enterprise and Team customers.
- The tool reasons about code (traces data flows, reads commit histories, evaluates risky paths) rather than pattern-matching against known vulnerability databases.
- Cybersecurity stocks fell sharply: CrowdStrike down ~8%, Cloudflare over 8%.
- Skepticism from security researchers: no published false positive rates or cost data. "Marketing-first, science-second."
- **myCode positioning:** Claude Code Security finds *security* vulnerabilities through reasoning. myCode finds *reliability* vulnerabilities through dependency-aware stress testing. Complementary, not competing. Same thesis — AI can find what traditional tools miss — different attack surface.
- **Investor framing:** "Anthropic just validated that AI can find what traditional tools miss. They're doing it for security. We're doing it for reliability."
- Source: https://venturebeat.com/security/anthropic-claude-code-security-reasoning-vulnerability-hunting

### Data Point 17: DOD AI Coding Tools Solicitation (DefenseScoop, Feb 26, 2026)
- DOD's CDAO, in partnership with the Army, seeking AI-enabled coding tools for "tens of thousands" of military and civilian developers.
- Both IDE-based coding assistants and CLI-based agentic coders in scope.
- Must be deployable at edge, in air-gapped networks, at IL5 security clearance (FedRAMP High + DISA IL5 PA).
- Built-in attribution and traceability mechanisms required — tagging which code is AI-generated.
- Submission deadline: March 6, 2026.
- **Gap identified:** Solicitation addresses code generation. No mention of code verification. Same pattern as commercial market — generation first, verification missing.
- **myCode positioning:** myCode already runs --offline with zero API calls. Air-gapped readiness is closer than most AI tools. Attribution metadata could be consumed by myCode to prioritise AI-generated files for stress testing.
- **Investor framing:** "The US Department of Defense is deploying AI coding tools to tens of thousands of developers. Nobody is soliciting the verification layer yet."
- Source: https://defensescoop.com/2026/02/26/dod-wants-ai-enabled-coding-tools-for-developer-workforce/

### Data Point 18: Columbia University — Vibe Coding Security Debt Crisis (Towards Data Science, Feb 22, 2026)
- Columbia University research team evaluated top coding agents and vibe coding tools. Published as TDS Editor's Pick.
- Three systematic failure patterns identified:
  1. **Speed over safety:** Agents remove validation checks, relax database policies, disable authentication flows to resolve runtime errors.
  2. **Unaware of side effects:** Agent fixes bug in one file, causes breaking changes or security leaks in referencing files.
  3. **Pattern matching, not judgement:** LLMs don't understand why security checks exist. "To an AI, a security wall is just a bug preventing the code from running."
- **Moltbook case study:** AI-agent social network built via vibe coding. Misconfigured Supabase database exposed 1.5 million API keys and 35,000 user email addresses to the public internet. Root cause: vibe coding shortcuts.
- Author's conclusion: "Coding agents optimize for making code run, not making code safe."
- References Wiz security report on Moltbook and Columbia's own "9 Critical Failure Patterns of Coding Agents" research.
- **Investor framing:** "Academic research from Columbia confirms: AI coding agents systematically remove safety guards to make code run. The Moltbook incident — 1.5M API keys exposed — is the inevitable consequence."
- Source: https://towardsdatascience.com/the-reality-of-vibe-coding-ai-agents-and-the-security-debt-crisis/
- Columbia research: https://daplab.cs.columbia.edu/general/2026/01/08/9-critical-failure-patterns-of-coding-agents.html

---

## Complete Evidence Table

| # | Source | Date | Category | Key Finding |
|---|--------|------|----------|-------------|
| 1 | FT — Oliver Roeder | Feb 23, 2026 | Market behaviour | No testing mentioned in vibe coding success story |
| 2 | Boris Cherny (Claude Code) | Feb 22–23, 2026 | Industry insider | "Failure modes may become foundational skill" |
| 3 | Anthropic — Skill Formation | Jan 29, 2026 | Academic research | AI users 17% worse at debugging |
| 4 | Anthropic — COBOL/IBM | Feb 23, 2026 | Enterprise impact | Legacy modernisation needs verification at scale |
| 5 | Anthropic — Agent Autonomy | Feb 22, 2026 | Usage data | 91% of AI code steps get no human review |
| 6 | InfoWorld — Matt Asay | Feb 23, 2026 | Market thesis | "True bottleneck is validation" |
| 7 | Dario Amodei | Feb 2026 | Industry leadership | AI coding acceleration confirmed |
| 8 | WSJ enforcement | Feb 2026 | Regulatory | Compliance demand for verification |
| 9 | JPost vibe coding | Feb 2026 | Market awareness | Mainstream media coverage |
| 10 | NDTV firing incident | Feb 2026 | Consequences | Career impact of unverified AI code |
| 11 | Check Point — Claude Code CVEs | Feb 25–26, 2026 | Security | RCE and API exfiltration in AI coding tools |
| 12 | VolodsTaimi — Lovable breach | Feb 26, 2026 | Security | 18,697 users exposed, inverted auth logic |
| 13 | Escape.tech scan | Oct 2025 | Security | 2,000+ vulns across 5,600 vibe-coded apps |
| 14 | The New Stack — Karpathy | Feb 26, 2026 | Market maturation | "Vibe coding" → "agentic engineering", verification gap explicit |
| 15 | LeadsNavi — geography | Feb 5, 2026 | Market sizing | Europe leads search volume, Middle East unmapped |
| 16 | Anthropic — Claude Code Security | Feb 24, 2026 | Validation | 500+ zero-days via AI reasoning; complementary to myCode |
| 17 | DOD solicitation | Feb 26, 2026 | Government | AI coding tools for tens of thousands, no verification layer |
| 18 | Columbia University / TDS | Feb 22, 2026 | Academic | Agents optimise for running, not safety; 1.5M API keys exposed |

---

## Additional Security Context

- Semafor (May 2025): Lovable security flaw unfixed for months
- Matt Palmer (Replit): 170 of 1,645 Lovable apps had same security flaw (March 2025)
- Daniel Asaria: Infiltrated multiple "top launched" Lovable sites in 47 minutes (April 2025)
- Tenzai: 5 vibe coding tools × 3 identical apps = 69 vulnerabilities, 6 critical
- Simon Willison: "This is the single biggest challenge with vibe coding."
- Gene Kim: Writing *Vibe Coding* book (DevOps pioneer — Phoenix Project, Unicorn Project — significant enterprise reach)

---

## What myCode Catches vs Security Scanners

- **myCode v1 catches:** Behavioural failures under stress — inverted auth logic under concurrent access, memory scaling, dependency interaction failures, timeout cascading. Semantic conformance testing and intent-behaviour divergence detection.
- **myCode v1 doesn't catch:** Exposed API keys, missing RLS policies, XSS from dangerouslySetInnerHTML, hardcoded secrets. These need security scanners.
- **Honest framing:** myCode addresses the behavioural verification gap. Security scanners address the configuration gap. Both are needed. Neither currently exists in a form non-engineers can use — that is the market opportunity.

---

*Machine Adjacent Systems (MAS) — ADGM, Abu Dhabi*
*This document compiles market evidence supporting the myCode product thesis.*
*Last updated: February 28, 2026 (Session 8)*
