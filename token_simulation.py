#!/usr/bin/env python3
"""
myCode — Token Consumption Simulation (Phase B)
Sends realistic myCode prompts to Gemini API to measure token volumes.
Projects costs for DeepSeek and Claude from measured token counts.

Usage:
    python3 token_simulation.py

Requires .env file with:
    GEMINI_API_KEY=...
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------------
# Load .env manually (no external dependency)
# ---------------------------------------------------------------------------
def load_env(path=".env"):
    if not Path(path).exists():
        print(f"ERROR: {path} not found. Create it with your API keys.")
        sys.exit(1)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ[key.strip()] = value.strip()

load_env()

GEMINI_KEY = os.environ.get("GEMINI_API_KEY")

if not GEMINI_KEY:
    print("ERROR: GEMINI_API_KEY not found in .env")
    sys.exit(1)

# ---------------------------------------------------------------------------
# API call helper
# ---------------------------------------------------------------------------
import urllib.request
import urllib.error

def call_gemini(messages, model="gemini-2.0-flash"):
    """Call Gemini API. Returns (response_text, usage_dict, elapsed_seconds)."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={GEMINI_KEY}"

    # Convert from OpenAI-style messages to Gemini format
    system_text = ""
    gemini_contents = []
    for m in messages:
        if m["role"] == "system":
            system_text = m["content"]
        elif m["role"] == "user":
            gemini_contents.append({"role": "user", "parts": [{"text": m["content"]}]})
        elif m["role"] == "assistant":
            gemini_contents.append({"role": "model", "parts": [{"text": m["content"]}]})

    body = {"contents": gemini_contents}
    if system_text:
        body["systemInstruction"] = {"parts": [{"text": system_text}]}
    body["generationConfig"] = {"maxOutputTokens": 2048}

    payload = json.dumps(body).encode("utf-8")

    req = urllib.request.Request(url, data=payload, headers={
        "Content-Type": "application/json",
    })

    start = time.time()
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        elapsed = time.time() - start
        text = data["candidates"][0]["content"]["parts"][0]["text"]
        usage = data.get("usageMetadata", {})
        return text, usage, elapsed
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        return f"HTTP ERROR {e.code}: {body}", {}, time.time() - start
    except Exception as e:
        return f"ERROR: {e}", {}, time.time() - start


# ---------------------------------------------------------------------------
# Realistic myCode prompts — one per LLM-calling component
# ---------------------------------------------------------------------------

PROMPTS = [
    {
        "name": "Conversational Interface",
        "description": "Extract operational intent from a non-technical user describing their project",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are myCode's conversational interface. Your job is to understand what "
                    "a non-technical user's project does, who it's for, and what conditions it "
                    "operates under. Ask clarifying questions in plain language. Do not use "
                    "engineering jargon. Extract: project purpose, expected user count, data "
                    "volume expectations, uptime requirements, and critical failure scenarios "
                    "the user cares about. Summarize your understanding in a structured format "
                    "at the end."
                ),
            },
            {
                "role": "user",
                "content": (
                    "I built an expense tracker for my small business. It's a Flask app with "
                    "a SQLite database. About 5 employees use it to log expenses and my "
                    "accountant downloads a CSV at the end of each month. It's hosted on a "
                    "$5 DigitalOcean droplet. Nothing fancy, just needs to work. Sometimes "
                    "people upload receipt photos too. I'm worried about what happens when "
                    "tax season comes and everyone dumps a month of expenses at once."
                ),
            },
        ],
    },
    {
        "name": "Scenario Generator",
        "description": "Generate stress test configurations from parsed project data + user intent",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are myCode's scenario generator. Given a parsed project summary and "
                    "the user's operational intent, generate specific stress test configurations. "
                    "For each test, specify: test category (data volume / memory profiling / "
                    "edge cases / concurrent execution / blocking I/O), exact parameters, "
                    "synthetic data requirements, expected resource consumption, and success/"
                    "failure criteria. Tests should target dependency interaction chains, not "
                    "individual components. Output as structured JSON."
                ),
            },
            {
                "role": "user",
                "content": json.dumps({
                    "project_summary": {
                        "framework": "Flask 3.0.2",
                        "database": "SQLite3 (stdlib)",
                        "orm": "SQLAlchemy 2.0.28",
                        "dependencies": ["pandas 2.2.1", "Pillow 10.2.0", "openpyxl 3.1.2"],
                        "entry_point": "app.py",
                        "routes": [
                            "POST /expense (creates expense record, optional file upload)",
                            "GET /expenses (lists all expenses, filterable by date/employee)",
                            "GET /export/csv (generates CSV download of all expenses)",
                            "POST /upload-receipt (saves receipt image, links to expense)",
                        ],
                        "database_tables": ["expenses", "employees", "receipts"],
                        "file_operations": ["receipt upload to ./uploads/", "CSV generation"],
                    },
                    "operational_intent": {
                        "users": 5,
                        "peak_scenario": "tax season — all 5 employees submitting a month of expenses simultaneously",
                        "data_volume": "~50-100 expenses per month normally, ~500 during tax crunch",
                        "hosting": "single $5 DigitalOcean droplet, 1GB RAM, 1 vCPU",
                        "critical_concern": "concurrent writes during tax season + large CSV export",
                        "uptime": "business hours, not mission-critical but embarrassing if down",
                    },
                }, indent=2),
            },
        ],
    },
    {
        "name": "Report Generator",
        "description": "Produce plain-language diagnostic report from raw stress test results",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are myCode's report generator. Given raw stress test execution results, "
                    "produce a plain-language diagnostic report for a non-technical user. "
                    "Explain what broke, at what point, and why it matters in terms of their "
                    "stated use case. Use degradation curves where relevant (describe them "
                    "textually). Do NOT suggest fixes or generate patches. Diagnose only. "
                    "Group findings by severity: critical (will break in production), warning "
                    "(will degrade under load), and info (worth knowing). Reference the user's "
                    "own description of their project when explaining impact."
                ),
            },
            {
                "role": "user",
                "content": json.dumps({
                    "test_results": [
                        {
                            "test": "concurrent_writes",
                            "category": "concurrent execution",
                            "result": "FAILURE at 3 simultaneous writes",
                            "error": "sqlite3.OperationalError: database is locked",
                            "detail": "SQLite cannot handle concurrent write operations. When 3 employees submit expenses at the same time, the 3rd request fails with a database lock error. Flask's default threading mode allows concurrent requests but SQLite serializes writes.",
                            "load_level": "3 concurrent POST /expense requests",
                            "memory_at_failure": "45MB",
                            "time_to_failure": "0.8 seconds",
                        },
                        {
                            "test": "csv_export_under_load",
                            "category": "data volume + blocking I/O",
                            "result": "DEGRADED at 500 records",
                            "detail": "CSV export loads all records into memory via pandas DataFrame. At 500 records with receipt metadata, response time is 4.2 seconds. At 2000 records, response time is 18.7 seconds and memory spikes to 380MB (approaching 1GB droplet limit). The route blocks the Flask process during generation, meaning no other requests can be served.",
                            "memory_profile": [
                                {"records": 100, "memory_mb": 52, "response_sec": 0.8},
                                {"records": 500, "memory_mb": 128, "response_sec": 4.2},
                                {"records": 1000, "memory_mb": 245, "response_sec": 9.1},
                                {"records": 2000, "memory_mb": 380, "response_sec": 18.7},
                            ],
                        },
                        {
                            "test": "receipt_upload_volume",
                            "category": "data volume + memory",
                            "result": "WARNING",
                            "detail": "Receipt images stored in ./uploads/ with no size limit enforced. A 15MB image upload succeeds but causes memory spike to 210MB during Pillow processing. No cleanup of temporary files. Disk usage grows linearly with no pruning strategy.",
                            "memory_spike": "210MB on 15MB upload",
                        },
                        {
                            "test": "edge_case_inputs",
                            "category": "edge cases",
                            "result": "FAILURE",
                            "detail": "Empty string in expense amount field causes unhandled ValueError in float() conversion. Negative amounts accepted without validation. Date field accepts future dates. SQL injection not tested (out of scope for v1).",
                            "failures": [
                                "Empty amount → ValueError (unhandled, returns 500)",
                                "Negative amount → accepted, corrupts totals",
                                "Future date → accepted",
                                "Amount = 'abc' → ValueError (unhandled, returns 500)",
                            ],
                        },
                    ],
                    "user_context": {
                        "project": "expense tracker for small business",
                        "peak_concern": "tax season crunch with all 5 employees submitting at once",
                        "hosting": "1GB RAM DigitalOcean droplet",
                    },
                }, indent=2),
            },
        ],
    },
]


# ---------------------------------------------------------------------------
# Cost projection using published pricing
# ---------------------------------------------------------------------------

PRICING = {
    "deepseek-chat":  {"input": 0.14,  "output": 0.28},
    "claude-sonnet":  {"input": 3.00,  "output": 15.00},
    "claude-opus":    {"input": 15.00, "output": 75.00},
    "gemini-flash":   {"input": 0.10,  "output": 0.40},
}


def calc_cost(input_tokens, output_tokens, model_key):
    prices = PRICING.get(model_key, {"input": 0, "output": 0})
    input_cost = (input_tokens / 1_000_000) * prices["input"]
    output_cost = (output_tokens / 1_000_000) * prices["output"]
    return input_cost + output_cost


# ---------------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------------

def run_simulation():
    print("=" * 70)
    print("myCode — Token Consumption Simulation")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print("\nMethod: Gemini API for actual token measurement.")
    print("DeepSeek and Claude costs projected from measured token counts.\n")

    results = []
    total_input = 0
    total_output = 0

    for prompt in PROMPTS:
        print(f"{'─' * 70}")
        print(f"TEST: {prompt['name']}")
        print(f"  {prompt['description']}")
        print(f"{'─' * 70}")

        print(f"\n  [Gemini 2.0 Flash] Sending request...", end=" ", flush=True)
        text, usage, elapsed = call_gemini(prompt["messages"])
        is_error = text.startswith("ERROR") or text.startswith("HTTP ERROR")
        print(f"{'FAILED' if is_error else 'OK'} ({elapsed:.1f}s)")

        if is_error:
            print(f"    Error: {text[:300]}")
            input_tokens = 0
            output_tokens = 0
        else:
            input_tokens = usage.get("promptTokenCount", 0)
            output_tokens = usage.get("candidatesTokenCount", 0)
            total_input += input_tokens
            total_output += output_tokens

            print(f"    Input tokens:  {input_tokens:,}")
            print(f"    Output tokens: {output_tokens:,}")
            print(f"    Response time: {elapsed:.1f}s")
            print(f"\n    Projected cost for this component:")
            for model_key, label in [("deepseek-chat", "DeepSeek"), ("claude-sonnet", "Claude Sonnet"), ("claude-opus", "Claude Opus")]:
                cost = calc_cost(input_tokens, output_tokens, model_key)
                print(f"      {label:20s} ${cost:.6f}")

        results.append({
            "component": prompt["name"],
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "response_time_sec": round(elapsed, 2),
            "response_text": text if not is_error else None,
            "error": text if is_error else None,
        })

    # --- Summary ---
    print(f"\n{'=' * 70}")
    print("SUMMARY — Per Session (all 3 components)")
    print(f"{'=' * 70}")

    print(f"\n  Total tokens measured: {total_input:,} input / {total_output:,} output")

    print(f"\n  Projected cost per session:")
    print(f"  {'Model':<25s} {'Input Cost':>12s} {'Output Cost':>12s} {'Total':>12s}")
    print(f"  {'─' * 61}")

    session_costs = {}
    for model_key, label in [("deepseek-chat", "DeepSeek V3"), ("claude-sonnet", "Claude Sonnet 4.5"), ("claude-opus", "Claude Opus 4.6"), ("gemini-flash", "Gemini 2.0 Flash")]:
        prices = PRICING[model_key]
        in_cost = (total_input / 1_000_000) * prices["input"]
        out_cost = (total_output / 1_000_000) * prices["output"]
        total_cost = in_cost + out_cost
        session_costs[model_key] = total_cost
        print(f"  {label:<25s} ${in_cost:>10.6f} ${out_cost:>10.6f} ${total_cost:>10.6f}")

    print(f"\n  Projected monthly cost at various volumes:")
    print(f"  {'Model':<25s} {'100 sessions':>14s} {'1,000 sessions':>14s} {'10,000 sessions':>14s}")
    print(f"  {'─' * 67}")
    for model_key, label in [("deepseek-chat", "DeepSeek V3"), ("claude-sonnet", "Claude Sonnet 4.5"), ("claude-opus", "Claude Opus 4.6")]:
        c = session_costs[model_key]
        print(f"  {label:<25s} ${c * 100:>12.2f} ${c * 1000:>12.2f} ${c * 10000:>12.2f}")

    if session_costs.get("deepseek-chat", 0) > 0:
        ratio = session_costs["claude-sonnet"] / session_costs["deepseek-chat"]
        print(f"\n  Cost ratio: Claude Sonnet is {ratio:.0f}x more expensive than DeepSeek")

    # --- Save full results ---
    output = {
        "simulation_date": datetime.now().isoformat(),
        "method": "Gemini 2.0 Flash for token measurement, costs projected from published pricing",
        "token_totals": {
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
        },
        "projected_session_costs": {k: round(v, 6) for k, v in session_costs.items()},
        "projected_monthly_1000_sessions": {k: round(v * 1000, 2) for k, v in session_costs.items()},
        "pricing_used": PRICING,
        "results": results,
    }

    output_path = "simulation_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Full results (including response texts) saved to: {output_path}")
    print(f"  Review response quality in that file to assess the quality gap.")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    run_simulation()
