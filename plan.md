# Plan: Gemini Integration — LLM-Enhanced Fix Prompts

**STATUS: AWAITING REVIEW**

Scope: When a finding has `source_file` and `source_function` populated, pass the function's source code to Gemini 2.0 Flash with the diagnostic finding as context, get a line-specific fix suggestion back, and append it to the deterministic fix prompt. Silent fallback to deterministic-only on any failure.

---

## 1. Where Fix Prompt Generation Happens Today

**Call chain:**

```
ReportGenerator.generate()                     # report.py:1058
  → assembles Finding objects with _tag_source  # report.py:1342
  → stores in DiagnosticReport.findings

DiagnosticReport.to_dict()                     # report.py:684
  → _finding_dict(f)                            # report.py:684
    → generate_finding_prompt(f)                # documents.py:1025

documents.py PDF/markdown generation
  → generate_finding_prompt(f)                  # documents.py:869, 3289
```

`generate_finding_prompt()` (documents.py:1025) takes a `Finding` and builds a deterministic prompt from:
- `f.severity`, `f.title`
- `f.source_file`, `f.source_function` (names only — no code)
- `f.affected_dependencies`
- `f._load_level`
- `_build_diagnosis(f)` — pattern-matched architecture-aware diagnosis
- `_build_fix(f)` — pattern-matched fix objective

**Key observation:** At prompt generation time, neither `IngestionResult` nor the project path are available. Only the `Finding` dataclass fields are accessible.

---

## 2. How to Read the Function Body

**Option A (recommended): Use IngestionResult at enrichment time.**

`IngestionResult` has:
- `project_path: str` — absolute path to the project copy (inside session venv)
- `file_analyses: list[FileAnalysis]` — each with `functions: list[FunctionInfo]`

`FunctionInfo` has:
- `file_path: str` — relative to project root
- `name: str` — function name
- `lineno: int` — start line (1-indexed)
- `end_lineno: int` — end line (inclusive)

**Extraction method:**
1. Match `Finding.source_file` → `FileAnalysis.file_path`
2. Match `Finding.source_function` → `FunctionInfo.name` within that file
3. Read `project_path / file_path`, extract lines `lineno` through `end_lineno`
4. Cap at 80 lines (truncate with `# ... truncated` comment). This keeps the Gemini prompt under ~3K tokens for function body.

**Why not re-parse?** The ingester already parsed the AST and stored line ranges. Re-reading is just `readlines()[lineno-1:end_lineno]`. No re-parsing needed.

**Edge cases:**
- `source_function` not found in `file_analyses` → skip enrichment (silent fallback)
- `end_lineno == 0` (JS regex extraction doesn't always set it) → skip enrichment
- File no longer readable (session cleanup race) → skip enrichment
- Function body > 80 lines → truncate, note truncation in prompt

---

## 3. Gemini API Integration

**Endpoint:** `https://generativelanguage.googleapis.com/v1beta/openai/` — already defined in the codebase as `GEMINI_BASE_URL` in `scenario.py:180`.

**Model:** `gemini-2.0-flash`

**Configuration:**

New environment variable: `GEMINI_API_KEY` (already used by the existing LLM pipeline — same key reused).

No new config class. Create a dedicated `LLMConfig` instance for fix-prompt enrichment:

```python
_fix_llm_config = LLMConfig(
    api_key=api_key,           # from existing pipeline config
    base_url=GEMINI_BASE_URL,  # same as scenario generator
    model="gemini-2.0-flash",  # hardcoded — this is a cheap, fast call
    max_tokens=256,            # fix suggestions are short
    temperature=0.2,           # low creativity, high precision
    timeout_seconds=8.0,       # fast fail — this is an enhancement, not critical path
    max_retries=1,             # single retry only
)
```

**Why `gemini-2.0-flash` and not `gemini-2.5-flash`?** 2.0 Flash is cheaper and faster. Fix suggestions don't need deep reasoning — they need pattern recognition on a short function with diagnostic context already provided. If quality proves insufficient, upgrade to 2.5 Flash later.

**Why separate LLMConfig?** The scenario generator config may use a different model (e.g., BYOK, Sonnet for freemium). Fix enrichment always uses Gemini 2.0 Flash at low cost, regardless of what the user's main LLM config is. If `GEMINI_API_KEY` is not set and the user is using BYOK, fix enrichment is silently skipped.

---

## 4. The Prompt to Gemini

**System message:**

```
You are a code diagnostic assistant for myCode, a stress-testing tool. Given a function's source code and a diagnostic finding from a stress test, provide a specific, line-referenced fix suggestion.

Rules:
- Reference specific line numbers or variable/function names from the code.
- Be concise: 2-4 sentences maximum.
- Be actionable: say exactly what to change and where.
- Do not wrap output in markdown fences or formatting.
- Do not repeat the diagnosis — only provide the fix.
- If the code does not clearly relate to the finding, reply with exactly: NO_SUGGESTION
```

**User message template:**

```
Finding: [{severity}] {title}
File: {source_file} → {source_function}()
Diagnosis: {diagnosis}
Dependencies involved: {affected_dependencies}
{load_level_line}

Source code ({source_file}, lines {lineno}-{end_lineno}):
```{language}
{function_body}
```

What specific code change fixes this?
```

**Output constraints:**
- Max 256 tokens (enforced via `max_tokens`)
- Response must be < 100 words (validated post-response — reject if over)
- `NO_SUGGESTION` response → silent fallback
- Empty response → silent fallback

---

## 5. Silent Fallback Conditions

Every condition below results in the deterministic prompt being returned unchanged — no error shown to user, no degraded output. The Gemini enhancement is invisible when it fails.

| # | Condition | Check point |
|---|-----------|-------------|
| 1 | `GEMINI_API_KEY` not set and no LLM config available | Before attempting call |
| 2 | `source_file` or `source_function` empty on Finding | Before function lookup |
| 3 | Function not found in `file_analyses` | During FunctionInfo lookup |
| 4 | `end_lineno == 0` (line range unavailable) | During FunctionInfo lookup |
| 5 | Source file not readable from disk | During file read |
| 6 | Function body is empty (0 lines) | After extraction |
| 7 | Gemini call times out (> 8 seconds) | During API call |
| 8 | Gemini returns HTTP error (4xx, 5xx) | During API call |
| 9 | Gemini returns empty content | After API call |
| 10 | Response is `NO_SUGGESTION` | After API call |
| 11 | Response exceeds 100 words | After API call |
| 12 | Response contains error indicators (`"error"`, `"I cannot"`, `"I'm sorry"`) | After API call |
| 13 | Any uncaught exception in the enrichment path | Outer try/except |

**Logging:** All fallbacks logged at `DEBUG` level with reason. No `WARNING` or `ERROR` — this is expected behavior, not a failure.

---

## 6. Where to Insert in the Pipeline

**Proposed approach: Enrichment step in `ReportGenerator.generate()`, NOT inside `generate_finding_prompt()`.**

**Rationale:**
- `generate_finding_prompt()` is a pure function that takes only a `Finding`. Passing `IngestionResult`, `LLMConfig`, and the project path into it would bloat its signature and break its deterministic contract.
- `ReportGenerator.generate()` already has access to `ingestion` (with `project_path` and `file_analyses`) and `self._llm_config` (with the API key).
- Enrichment happens once per finding, before any prompt generation call.

**Implementation location:** `src/mycode/report.py`, new private method `_enrich_finding_with_llm_fix()`.

**Call site:** After the finding assembly loop (after line ~1354 in report.py), before `to_dict()` is ever called. New loop:

```python
# After all findings assembled, enrich with LLM fix suggestions
for f in report.findings:
    self._enrich_finding_with_llm_fix(f, ingestion)
```

**New field on Finding:** `llm_fix_suggestion: str = ""` — populated by enrichment, empty string when not enriched.

**Integration with `generate_finding_prompt()`:** After the existing `Fix:` line, append:

```python
if f.llm_fix_suggestion:
    parts.append(f"Suggested fix: {f.llm_fix_suggestion}")
```

**New helper module:** `src/mycode/fix_enrichment.py` — contains:
- `extract_function_body(ingestion: IngestionResult, source_file: str, source_function: str) -> tuple[str, int, int]` — returns `(body, lineno, end_lineno)` or `("", 0, 0)` on failure
- `get_llm_fix_suggestion(finding: Finding, function_body: str, lineno: int, end_lineno: int, llm_config: LLMConfig, language: str) -> str` — returns suggestion or `""` on any failure
- All fallback logic encapsulated here. report.py stays clean.

**Why a separate module?** Keeps LLM call logic out of both report.py (already 1500+ lines) and documents.py (already 3000+ lines). Single responsibility. Easy to test in isolation.

---

## 7. Public Caveat Text

When at least one finding in the report has a non-empty `llm_fix_suggestion`, append this caveat to both the PDF and JSON outputs:

**In the PDF (Understanding Your Results), in the "About this report" section:**

> Some fix suggestions in this report were generated by an AI model (Gemini) based on your source code and the diagnostic findings. These suggestions are starting points, not guaranteed fixes. Always review AI-generated suggestions in the context of your full codebase before applying them.

**In the JSON (for Coding Agent), as a top-level field:**

```json
{
  "llm_fix_caveat": "Fix suggestions marked 'Suggested fix' were generated by Gemini based on the function source code and diagnostic context. Treat as starting points — verify before applying."
}
```

**When no LLM suggestions were generated:** Omit both. No mention of Gemini in reports that didn't use it.

---

## 8. Testing Approach

**Unit tests (no real API calls):**

**A. Function body extraction** (`test_fix_enrichment.py`):
- Test `extract_function_body()` with a mock `IngestionResult` containing known `FileAnalysis` + `FunctionInfo`
- Write a temporary Python file, populate `FunctionInfo` with correct line ranges, verify extraction
- Test edge cases: function not found, `end_lineno == 0`, file not readable, body > 80 lines (truncation)

**B. LLM suggestion with mocked backend** (`test_fix_enrichment.py`):
- Patch `LLMBackend.generate` to return a canned `LLMResponse`
- Verify prompt template is correctly assembled
- Verify response is accepted when valid (< 100 words, not `NO_SUGGESTION`)
- Verify silent fallback on each condition from Section 5:
  - Mock returning `NO_SUGGESTION`
  - Mock returning 150-word response (rejected)
  - Mock returning empty string
  - Mock raising `LLMError` (timeout/HTTP error)
  - Mock raising unexpected exception
  - Test with `api_key=None` (no call made)

**C. Integration with Finding** (`test_fix_enrichment.py`):
- Create a Finding with `source_file` + `source_function`
- Run enrichment with mocked backend
- Verify `f.llm_fix_suggestion` is populated
- Verify `generate_finding_prompt(f)` includes the suggestion line

**D. End-to-end prompt output** (in existing `test_documents.py`):
- Verify `generate_finding_prompt()` output includes `Suggested fix:` when `llm_fix_suggestion` is set
- Verify it does NOT include the line when `llm_fix_suggestion` is empty

**No test requires a real Gemini API call.** All LLM interaction is behind `LLMBackend.generate()`, which is trivially mockable via `unittest.mock.patch`.

---

## Files Changed

| File | Change |
|------|--------|
| `src/mycode/fix_enrichment.py` | **NEW** — function body extraction + LLM fix suggestion logic |
| `src/mycode/report.py` | Add `llm_fix_suggestion` field to `Finding`, call enrichment loop in `generate()` |
| `src/mycode/documents.py` | Append `Suggested fix:` line in `generate_finding_prompt()`, add caveat to PDF/JSON |
| `tests/test_fix_enrichment.py` | **NEW** — unit tests for extraction, suggestion, fallback |
| `tests/test_documents.py` | Add tests for prompt output with/without LLM suggestion |

---

## Not In Scope

- Freemium/enterprise LLM routing (this is free-tier Gemini only)
- Caching LLM responses across runs
- Passing `installed_version`, `call_graph`, or `is_outdated` into prompts (separate issue per MEMORY.md)
- JavaScript function body extraction (JS `end_lineno` is unreliable — enrichment silently skips these via fallback condition #4)
- User-facing toggle to disable Gemini enrichment (silent by design — no config needed)
