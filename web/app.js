/* myCode Web Interface — Application Logic */

/* global MYCODE_CONFIG */

const API = MYCODE_CONFIG.API_URL;

let currentJobId = null;
let converseTurn = 1;
let pollTimer = null;
let elapsedTimer = null;
let elapsedStart = null;

// ── Helpers ──

function $(id) { return document.getElementById(id); }

function show(id) { $(id).classList.remove("hidden"); }
function hide(id) { $(id).classList.add("hidden"); }

function escapeHtml(str) {
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
}

function highlightConsequence(text, severity) {
    // Split into sentences, highlight consequence sentences in red.
    const sentences = text.match(/[^.!]+[.!]+/g) || [text];
    let matched = false;
    const parts = sentences.map((s, i) => {
        const lower = s.trimStart().toLowerCase();
        const isConsequence = lower.startsWith("this means") ||
            lower.startsWith("in practice,") ||
            lower.includes("your app will") ||
            lower.includes("your app's") ||
            lower.includes("memory usage grows") ||
            lower.includes("showed failures in");
        if (isConsequence) {
            matched = true;
            return { html: `<span class="consequence">${escapeHtml(s)}</span>` };
        }
        return { html: escapeHtml(s), idx: i };
    });
    // Warning fallback: if no consequence matched, highlight last sentence
    if (!matched && severity === "warning" && sentences.length > 1) {
        const last = sentences.length - 1;
        parts[last] = { html: `<span class="consequence">${escapeHtml(sentences[last])}</span>` };
    }
    return parts.map(p => p.html).join("");
}

async function apiPost(path, formData) {
    const res = await fetch(`${API}${path}`, { method: "POST", body: formData });
    return res.json();
}

async function apiGet(path) {
    const res = await fetch(`${API}${path}`);
    return res.json();
}

function formatElapsed(seconds) {
    const s = Math.floor(seconds);
    if (s < 60) return s + "s";
    const m = Math.floor(s / 60);
    const rem = s % 60;
    if (m < 60) return m + "m " + rem + "s";
    const h = Math.floor(m / 60);
    return h + "h " + (m % 60) + "m";
}

function startElapsedTimer() {
    elapsedStart = Date.now();
    $("progress-elapsed").textContent = "0s";
    elapsedTimer = setInterval(() => {
        const seconds = (Date.now() - elapsedStart) / 1000;
        $("progress-elapsed").textContent = formatElapsed(seconds);
    }, 1000);
}

function stopElapsedTimer() {
    if (elapsedTimer) {
        clearInterval(elapsedTimer);
        elapsedTimer = null;
    }
}

// ── Input ──

async function submitUrl() {
    const url = $("github-url").value.trim();
    if (!url) return;
    $("go-btn").disabled = true;
    $("go-btn").textContent = "...";
    const fd = new FormData();
    fd.append("github_url", url);
    await runPreflight(fd);
}

async function submitFile(input) {
    const file = input.files[0];
    if (!file) return;
    $("upload-label").textContent = file.name;
    const fd = new FormData();
    fd.append("file", file);
    await runPreflight(fd);
}

// ── Preflight ──

async function runPreflight(formData) {
    show("preflight-section");
    show("preflight-loading");
    hide("preflight-content");

    try {
        const data = await apiPost("/api/preflight", formData);

        hide("preflight-loading");

        if (data.error) {
            $("preflight-content").innerHTML =
                `<div class="diagnostics-grid" id="diag-grid"></div>` +
                `<div id="viability-banner" class="viability-banner fail">${escapeHtml(data.error)}</div>`;
            show("preflight-content");
            resetInput();
            return;
        }

        currentJobId = data.job_id;
        renderPreflight(data);
        show("preflight-content");

        if (data.viability && !data.viability.viable) {
            showViabilityWarning(data.viability);
        }
        beginConversation();
    } catch (err) {
        hide("preflight-loading");
        $("preflight-content").innerHTML =
            `<div class="diagnostics-grid" id="diag-grid"></div>` +
            `<div id="viability-banner" class="viability-banner fail">Connection failed: ${escapeHtml(err.message)}</div>`;
        show("preflight-content");
        resetInput();
    }
}

function resetInput() {
    $("go-btn").disabled = false;
    $("go-btn").textContent = "Go";
}

function renderPreflight(data) {
    const v = data.viability || {};

    const pct = (val) => Math.round((val || 0) * 100);
    const cls = (val, threshold) => val >= threshold ? "pass" : val >= threshold * 0.6 ? "warn" : "fail";

    const bannerClass = v.viable ? "pass" : "fail";
    const bannerText = v.viable
        ? "Ready for testing"
        : escapeHtml(v.reason || "Baseline viability check failed");

    $("preflight-content").innerHTML = `
        <div class="diagnostics-grid" id="diag-grid">
            <div class="diag-item">
                <span class="diag-label">Language</span>
                <span class="diag-value">${escapeHtml(data.language || "unknown")}</span>
            </div>
            <div class="diag-item">
                <span class="diag-label">Project</span>
                <span class="diag-value">${escapeHtml(data.project_name || "—")}</span>
            </div>
            <div class="diag-item">
                <span class="diag-label">Dependencies installed</span>
                <span class="diag-value ${cls(v.install_rate, 0.5)}">${pct(v.install_rate)}%</span>
            </div>
            <div class="diag-item">
                <span class="diag-label">Imports working</span>
                <span class="diag-value ${cls(v.import_rate, 0.5)}">${pct(v.import_rate)}%</span>
            </div>
            <div class="diag-item">
                <span class="diag-label">Syntax valid</span>
                <span class="diag-value ${cls(v.syntax_rate, 0.25)}">${pct(v.syntax_rate)}%</span>
            </div>
            <div class="diag-item">
                <span class="diag-label">Library profiles</span>
                <span class="diag-value">${(data.profile_matches || []).length} matched</span>
            </div>
        </div>
        <div id="viability-banner" class="viability-banner ${bannerClass}">${bannerText}</div>
    `;
}

// ── Conversation ──

function showViabilityWarning(viability) {
    const pct = Math.round((viability.install_rate || 0) * 100);
    let msg = `Warning: only ${pct}% of dependencies installed.`;
    if (viability.reason) {
        msg += " " + viability.reason;
    }
    msg += " Test results may be limited.";

    const banner = $("viability-banner");
    if (banner) {
        banner.className = "viability-banner warn";
        banner.textContent = msg;
    }
}

function beginConversation() {
    converseTurn = 1;
    show("converse-section");
    requestConverseTurn(1, "");
}

async function requestConverseTurn(turn, userResponse) {
    const fd = new FormData();
    fd.append("job_id", currentJobId);
    fd.append("turn", turn.toString());
    fd.append("user_response", userResponse);

    const data = await apiPost("/api/converse", fd);

    if (data.error) {
        addMessage("system", "Error: " + data.error);
        return;
    }

    if (data.question) {
        addMessage("system", data.question);
    }

    if (data.done) {
        // Conversation complete — show run button
        if (data.constraints) {
            const summary = formatConstraints(data.constraints);
            if (summary) {
                addMessage("system", "Understood. " + summary);
            }
        }
        hide("reply-area");
        show("run-section");
    } else {
        converseTurn = data.turn + 1;
        show("reply-area");
        $("reply-input").value = "";
        $("reply-input").focus();
    }
}

function sendReply() {
    const text = $("reply-input").value.trim();
    if (!text) return;
    addMessage("user", text);
    hide("reply-area");
    requestConverseTurn(converseTurn, text);
}

// Handle Enter key in reply textarea
document.addEventListener("DOMContentLoaded", () => {
    const replyInput = $("reply-input");
    if (replyInput) {
        replyInput.addEventListener("keydown", (e) => {
            if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                sendReply();
            }
        });
    }

    const urlInput = $("github-url");
    if (urlInput) {
        urlInput.addEventListener("keydown", (e) => {
            if (e.key === "Enter") {
                e.preventDefault();
                submitUrl();
            }
        });
    }
});

function addMessage(role, text) {
    const thread = $("converse-thread");
    const div = document.createElement("div");
    div.className = `message ${role}`;
    div.textContent = text;
    thread.appendChild(div);
    div.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function formatConstraints(c) {
    const parts = [];
    if (c.user_scale) parts.push(`${c.user_scale} concurrent users`);
    if (c.usage_pattern) parts.push(c.usage_pattern.replace("_", " ") + " usage");
    if (c.deployment_context) parts.push(c.deployment_context.replace("_", " "));
    if (c.data_type) parts.push(c.data_type.replace("_", " ") + " data");
    if (c.analysis_depth) {
        const labels = {quick: "quick scan", standard: "standard analysis", deep: "deep analysis"};
        parts.push(labels[c.analysis_depth] || c.analysis_depth);
    } else if (c.timeout_per_scenario) {
        parts.push(`${c.timeout_per_scenario}s per test`);
    }
    return parts.length ? "Testing for: " + parts.join(", ") + "." : "";
}

// ── Analysis ──

async function startAnalysis() {
    $("run-btn").disabled = true;
    $("run-btn").textContent = "Starting...";
    show("progress-container");

    const fd = new FormData();
    fd.append("job_id", currentJobId);
    fd.append("offline", "true");

    const data = await apiPost("/api/analyze", fd);

    if (data.error) {
        $("progress-text").textContent = "Error: " + data.error;
        return;
    }

    hide("run-btn");
    startElapsedTimer();
    pollProgress();
}

function pollProgress() {
    pollTimer = setInterval(async () => {
        try {
            const data = await apiGet(`/api/status/${currentJobId}`);
            updateProgress(data);

            if (data.status === "completed") {
                clearInterval(pollTimer);
                stopElapsedTimer();
                fetchReport();
            } else if (data.status === "failed") {
                clearInterval(pollTimer);
                stopElapsedTimer();
                $("progress-text").textContent = "Failed: " + (data.error || "Unknown error");
                $("progress-fill").style.background = "var(--red)";
            }
        } catch (err) {
            // Network error — keep polling
        }
    }, 3000);
}

function updateProgress(data) {
    const p = data.progress;
    if (!p) return;

    const pct = p.progress_pct || 0;
    $("progress-fill").style.width = pct + "%";
    $("progress-pct").textContent = pct + "%";
    $("progress-text").textContent = `${p.scenarios_complete}/${p.scenarios_total} scenarios`;

    if (p.current_scenario) {
        $("progress-scenario").textContent = p.current_scenario.replace(/_/g, " ");
    }
}

// ── Report ──

async function fetchReport() {
    $("progress-text").textContent = "Generating report...";

    const data = await apiGet(`/api/report/${currentJobId}`);

    if (data.error) {
        $("progress-text").textContent = "Error: " + data.error;
        return;
    }

    // Replace progress bar with run summary
    const scenarioCount = (data.pipeline_summary && data.pipeline_summary.scenarios_run) || 0;
    const elapsed = elapsedStart ? formatElapsed((Date.now() - elapsedStart) / 1000) : "";
    const summaryParts = [];
    if (elapsed) summaryParts.push(elapsed);
    if (scenarioCount) summaryParts.push(scenarioCount + " scenario" + (scenarioCount !== 1 ? "s" : ""));
    const summaryText = summaryParts.length
        ? "Scan completed in " + summaryParts.join(" across ")
        : "Scan completed";
    $("progress-fill").style.width = "100%";
    $("progress-fill").style.background = "var(--green)";
    $("progress-pct").textContent = "";
    $("progress-elapsed").textContent = "";
    $("progress-text").textContent = summaryText;
    $("progress-scenario").textContent = "";

    show("report-section");
    renderReport(data.report, data.pipeline_summary, data.understanding_md, data.fixes_md, data.edition, data.has_pdf);
}

function renderReport(report, summary, understandingMd, fixesMd, edition, hasPdf) {
    if (!report) return;

    const content = $("report-content");
    // Store edition documents for download
    content.dataset.understandingMd = understandingMd || "";
    content.dataset.edition = edition || "0";
    content.dataset.hasPdf = hasPdf ? "true" : "";
    let html = "";

    // Stats
    const incompleteCount = summary.scenarios_incomplete || 0;
    html += `<div class="report-stats">
        <div class="stat"><span class="stat-value">${summary.scenarios_run || 0}</span><span class="stat-label">scenarios</span></div>
        <div class="stat"><span class="stat-value" style="color:var(--green)">${summary.scenarios_passed || 0}</span><span class="stat-label">passed</span></div>
        <div class="stat"><span class="stat-value" style="color:var(--red)">${summary.scenarios_failed || 0}</span><span class="stat-label">failed</span></div>
        ${incompleteCount > 0 ? `<div class="stat"><span class="stat-value" style="color:var(--amber)">${incompleteCount}</span><span class="stat-label">could not test</span></div>` : ""}
        <div class="stat"><span class="stat-value">${summary.total_errors || 0}</span><span class="stat-label">errors</span></div>
    </div>`;

    // Summary
    if (report.summary) {
        html += `<div class="report-summary">${escapeHtml(report.summary)}</div>`;
    }

    // Notes (confidence, data-type methodology)
    if (report.confidence_note) {
        html += `<div class="report-note">${escapeHtml(report.confidence_note)}</div>`;
    }
    if (report.data_type_note) {
        html += `<div class="report-note">${escapeHtml(report.data_type_note)}</div>`;
    }

    // Findings
    const findings = report.findings || [];
    if (findings.length > 0) {
        html += `<div class="section-title" style="margin-top:1.25rem">Findings</div>`;
        html += `<div class="findings-list">`;
        for (const f of findings) {
            html += renderFinding(f);
        }
        html += `</div>`;
    }

    // Degradation curves (JSON key is "degradation_curves")
    const degradations = report.degradation_curves || [];
    if (degradations.length > 0) {
        html += renderDegradations(degradations, findings);
    }

    // Incomplete tests
    const incomplete = report.incomplete_tests || [];
    if (incomplete.length > 0) {
        html += renderIncomplete(incomplete);
    }

    // Download buttons — two only
    const pdfLabel = hasPdf ? "PDF" : "Markdown";
    html += `<div class="download-row">
        <button class="btn btn-primary btn-sm" onclick="downloadUnderstanding()" style="border:2px solid var(--blue)">Understanding Your Results (${pdfLabel})</button>
        <button class="btn btn-secondary btn-sm" onclick="downloadJSON()" style="border:2px solid var(--border)">Download for Coding Agent (JSON)</button>
    </div>`;

    content.innerHTML = html;

    // Store report data for downloads
    content.dataset.report = JSON.stringify(report);
}

function renderFinding(f) {
    const severity = f.severity || "info";
    let html = `<div class="finding-card ${severity}">`;
    html += `<div class="finding-header">`;
    html += `<span class="severity-badge ${severity}">${severity}</span>`;
    html += `<span class="finding-title">${escapeHtml(f.title || "")}</span>`;
    html += `</div>`;

    if (f.description) {
        html += `<div class="finding-description">${highlightConsequence(f.description, severity)}</div>`;
    }
    if (f.details) {
        html += `<div class="finding-details">${escapeHtml(f.details)}</div>`;
    }
    if (f.affected_dependencies && f.affected_dependencies.length) {
        html += `<div class="finding-deps">Dependencies: ${f.affected_dependencies.map(escapeHtml).join(", ")}</div>`;
    }

    // Render grouped findings
    if (f.grouped_findings && f.grouped_findings.length > 0) {
        for (const gf of f.grouped_findings) {
            html += `<div style="margin-top:0.5rem;padding-left:0.75rem;border-left:2px solid var(--border)">`;
            html += `<div class="finding-description">${escapeHtml(gf.title || "")}: ${escapeHtml(gf.description || "")}</div>`;
            html += `</div>`;
        }
    }

    html += `</div>`;
    return html;
}

function _pl(n, singular, plural) {
    return n === 1 ? singular : (plural || singular + "s");
}

function humanizeStepLabel(label) {
    if (!label) return label;
    const s = String(label);
    let m, n;
    if ((m = s.match(/^data_size_(\d+)$/))) { n = Number(m[1]); return n.toLocaleString() + " " + _pl(n, "item"); }
    if ((m = s.match(/^concurrent_(\d+)$/))) { n = Number(m[1]); return n.toLocaleString() + " simultaneous " + _pl(n, "user"); }
    if ((m = s.match(/^api_concurrency_(\d+)$/))) { n = Number(m[1]); return n.toLocaleString() + " concurrent API " + _pl(n, "call"); }
    if ((m = s.match(/^wsgi_concurrent_(\d+)$/))) { n = Number(m[1]); return n.toLocaleString() + " concurrent " + _pl(n, "request"); }
    if ((m = s.match(/^async_handlers_(\d+)$/))) { n = Number(m[1]); return n.toLocaleString() + " async " + _pl(n, "handler"); }
    if ((m = s.match(/^async_load_(\d+)$/))) { n = Number(m[1]); return n.toLocaleString() + " concurrent " + _pl(n, "promise"); }
    if ((m = s.match(/^sync_threadpool_(\d+)$/))) { n = Number(m[1]); return n.toLocaleString() + " " + _pl(n, "thread") + " in pool"; }
    if ((m = s.match(/^sqlite_writers_(\d+)$/))) { n = Number(m[1]); return n.toLocaleString() + " concurrent " + _pl(n, "writer"); }
    if ((m = s.match(/^state_cycles_(\d+)$/))) { n = Number(m[1]); return n.toLocaleString() + " state mutation " + _pl(n, "cycle"); }
    if ((m = s.match(/^batch_(\d+)$/))) return Number(m[1]) === 0 ? "first iteration" : "iteration " + Number(m[1]).toLocaleString();
    if ((m = s.match(/^io_size_(\d+)$/))) {
        const size = Number(m[1]);
        if (size >= 1000000) return Math.round(size / 1000000) + "MB of data";
        if (size >= 1000) return Math.round(size / 1000) + "KB of data";
        return size.toLocaleString() + " bytes of data";
    }
    if ((m = s.match(/^gil_threads_(\d+)$/))) { n = Number(m[1]); return n + " parallel " + _pl(n, "thread"); }
    if ((m = s.match(/^compute_(\d+)$/))) { n = Number(m[1]); return n.toLocaleString() + " " + _pl(n, "item") + " of data"; }
    if ((m = s.match(/^rerun_rows_(\d+)$/))) { n = Number(m[1]); return n.toLocaleString() + " " + _pl(n, "row") + " of data"; }
    // HTTP load testing labels: "N concurrent"
    if ((m = s.match(/^(\d+) concurrent$/))) { n = Number(m[1]); return n.toLocaleString() + " concurrent " + _pl(n, "connection"); }
    return s.replace(/_/g, " ");
}

function humanizeMetricLabel(metric) {
    if (metric === "execution_time_ms") return "Response time under load";
    if (metric === "memory_peak_mb" || metric === "memory_mb" || metric === "memory_growth_mb") return "Memory usage under load";
    if (metric === "error_count") return "Errors under load";
    return "";
}

// Template descriptions — mirrors _TEMPLATE_DESCRIPTIONS in report.py
const _TEMPLATE_DESCRIPTIONS = {
    "concurrent_request_load": "Handling multiple users at once",
    "concurrent_session_load": "Handling multiple users at once",
    "large_payload_response": "Returning large results",
    "file_upload_scaling": "Handling file uploads",
    "file_upload_memory_stress": "Handling file uploads",
    "blocking_io_under_load": "Handling requests while waiting for data",
    "repeated_request_memory_profile": "Handling many requests over time",
    "script_rerun_cost": "Re-running with larger data",
    "cache_memory_growth": "Caching data over time",
    "repeated_interaction_memory_profile": "Extended user sessions",
    "large_download_memory": "Downloading large responses",
    "timeout_behavior": "Calling external APIs that respond slowly",
    "error_handling_resilience": "Dealing with unexpected API responses",
    "session_vs_individual_performance": "Making many API calls",
    "data_volume_scaling": "Processing larger amounts of data",
    "merge_memory_stress": "Combining large datasets",
    "iterrows_vs_vectorized": "Processing rows of data",
    "memory_profiling_over_time": "Repeated data operations over time",
    "edge_case_dtypes": "Handling unusual data formats",
    "concurrent_dataframe_access": "Accessing data from multiple places at once",
    "session_write_concurrency": "Multiple users writing at the same time",
    "array_size_scaling": "Working with larger arrays",
    "matrix_operation_scaling": "Heavy number crunching",
    "concurrent_array_access": "Accessing data from multiple threads",
    "edge_case_inputs": "Handling unusual or extreme inputs",
    "repeated_allocation_memory": "Allocating memory repeatedly",
    "async_concurrent_load": "Handling many requests at once",
    "sync_handler_thread_exhaustion": "Running synchronous code under load",
    "pydantic_validation_stress": "Validating request data at scale",
    "websocket_connection_scaling": "Handling many open connections",
    "large_response_streaming": "Streaming large responses",
    "middleware_chain_overhead": "Processing middleware layers",
    "async_error_handling": "Handling errors in async operations",
    "memory_under_load": "Managing memory under heavy traffic",
    "large_payload_handling": "Processing large request payloads",
};

function _humanizeIdentifier(raw) {
    return raw.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase());
}

function humanizeScenarioName(name) {
    if (!name) return name;
    const lower = name.toLowerCase();

    // HTTP endpoint scenarios
    if (lower === "http_get_root" || lower === "http_post_root") {
        return "Loading your application's main page";
    }
    if (lower.startsWith("http_get_") || lower.startsWith("http_post_")) {
        const parts = lower.split("_");
        if (parts.length >= 3) {
            const path = parts.slice(2).join("/");
            if (path && path !== "root") return "Loading the /" + path + " endpoint";
        }
    }

    // Template matching — try progressively shorter suffixes
    const parts = lower.split("_");
    for (let start = 1; start < parts.length; start++) {
        const key = parts.slice(start).join("_");
        if (_TEMPLATE_DESCRIPTIONS[key]) return _TEMPLATE_DESCRIPTIONS[key];
    }

    // Coupling scenario patterns
    if (lower.startsWith("coupling_compute_")) {
        const label = _humanizeIdentifier(name.slice("coupling_compute_".length));
        return "Running calculations (" + label + ")";
    }
    if (lower.startsWith("coupling_api_")) {
        const label = _humanizeIdentifier(name.slice("coupling_api_".length));
        return "Connecting to external services (" + label + ")";
    }
    if (lower.startsWith("coupling_render_")) {
        const label = _humanizeIdentifier(name.slice("coupling_render_".length));
        return "Rendering output (" + label + ")";
    }
    if (lower.startsWith("coupling_state_") || lower.startsWith("coupling_stateset")) {
        return "Updating shared state";
    }
    if (lower.startsWith("coupling_errorhandler_")) {
        return "Handling errors between components";
    }

    // Keyword fallbacks
    if (lower.includes("memory") || lower.includes("leak")) return "Managing memory over time";
    if (lower.includes("concurrent")) return "Handling simultaneous operations";
    if (lower.includes("scaling") || lower.includes("volume")) return "Processing larger amounts of data";

    // Check scenarios
    if (lower.endsWith("_check")) {
        const remainder = lower.replace(/_check$/, "").split("_").slice(1).join("_");
        if (remainder) return _humanizeIdentifier(remainder) + " behavior";
    }

    // Version discrepancy
    if (lower.endsWith("_version_discrepancy")) {
        const dep = lower.replace(/_version_discrepancy$/, "");
        return _humanizeIdentifier(dep) + " version compatibility";
    }

    // Generic stress
    if (lower.includes("generic_stress")) {
        return "General usage patterns";
    }

    // Last resort: strip dep prefix and humanize
    if (parts.length > 1) {
        return _humanizeIdentifier(parts.slice(1).join("_"));
    }
    const s = name.replace(/_/g, " ");
    return s.charAt(0).toUpperCase() + s.slice(1);
}

function stepLabel(step) {
    // Handle both {label, value} objects and [label, value] tuples
    if (step && typeof step === "object" && "label" in step) return step.label;
    if (Array.isArray(step)) return step[0];
    return "";
}

function stepValue(step) {
    if (step && typeof step === "object" && "value" in step) return step.value;
    if (Array.isArray(step)) return step[1];
    return 0;
}

function fmtCell(step, metric) {
    const val = formatMetricValue(stepValue(step), metric);
    const ctx = humanizeStepLabel(stepLabel(step));
    return val + " (" + ctx + ")";
}

function _findingSeverityForDp(d, findings) {
    /**
     * Match a degradation point against findings by category + metric.
     * Mirrors _finding_severity_for_dp in documents.py.
     */
    if (!findings || findings.length === 0) return null;

    const metric = (d.metric || "").toLowerCase();
    const isTime = metric.includes("time");
    const isMemory = metric.includes("memory");
    const dpName = (d.scenario_name || "").toLowerCase();

    const RESPONSE_KW = ["response time", "degradation", "latency"];
    const MEMORY_KW = ["memory"];

    let best = null;

    for (const f of findings) {
        if (f.severity !== "critical" && f.severity !== "warning") continue;

        let matched = false;
        const title = (f.title || "").toLowerCase();

        // HTTP category ↔ http_ scenario prefix + metric filter
        if (f.category === "http_load_testing" && dpName.startsWith("http_")) {
            if (isTime || isMemory) {
                const kws = isTime ? RESPONSE_KW : MEMORY_KW;
                if (kws.some(kw => title.includes(kw))) {
                    matched = true;
                } else {
                    // category matches but metric doesn't — fallback
                    if (!best || f.severity === "critical") {
                        best = best === "critical" ? "critical" : f.severity;
                    }
                    continue;
                }
            } else {
                matched = true;
            }
        }
        // Memory category ↔ memory metric
        else if (f.category === "memory_profiling" && isMemory) {
            matched = true;
        }
        // Category token in scenario name
        else {
            const TOKENS = {
                concurrent_execution: ["concurrent", "gil_contention"],
                data_volume_scaling: ["data", "volume", "scaling"],
                memory_profiling: ["memory"],
                blocking_io: ["blocking", "io"],
                edge_case_input: ["edge_case"],
                http_load_testing: ["http"],
            };
            const tokens = TOKENS[f.category] || [];
            if (tokens.some(t => dpName.includes(t))) {
                if (isTime && ["concurrent_execution", "blocking_io", "data_volume_scaling"].includes(f.category)) matched = true;
                else if (isMemory && f.category === "memory_profiling") matched = true;
                else if (["edge_case_input", "http_load_testing"].includes(f.category)) matched = true;
            }
        }

        // Dependency overlap fallback
        if (!matched && f.affected_dependencies) {
            for (const dep of f.affected_dependencies) {
                if (dpName.includes(dep.toLowerCase().replace(/-/g, "_"))) {
                    matched = true;
                    break;
                }
            }
        }

        if (matched) {
            if (f.severity === "critical") return "critical";
            best = "warning";
        }
    }
    return best;
}

function verdictForCurve(d, findings) {
    // Finding-based verdict takes precedence
    const findingSev = _findingSeverityForDp(d, findings);
    if (findingSev === "critical") return "Critical \u2014 see above";
    if (findingSev === "warning") return "Warning \u2014 see above";

    // Threshold-based fallback
    const steps = d.steps || [];
    if (steps.length === 0) return "";
    const lastVal = stepValue(steps[steps.length - 1]) || 0;
    const metric = (d.metric || "").toLowerCase();
    const isTime = metric.includes("time");
    const isMemory = metric.includes("memory");
    const isErrors = metric === "error_count";

    if (isTime) {
        if (lastVal < 100) return "No issues";
        if (lastVal < 500) {
            const bp = d.breaking_point ? humanizeStepLabel(d.breaking_point) : "";
            return bp ? "Noticeable above " + bp : "Noticeable at peak";
        }
        if (lastVal < 2000) return "Slow at peak load";
        return "Unresponsive at peak load";
    }
    if (isMemory) {
        if (lastVal < 50) return "Fine at your scale";
        if (lastVal < 200) return "Heavy \u2014 limits concurrent users";
        return "Very heavy \u2014 risk of crashes";
    }
    if (isErrors) {
        if (lastVal <= 0) return "No errors";
        return Math.round(lastVal) + " errors at peak";
    }
    return "";
}

function verdictColour(verdict) {
    const v = verdict.toLowerCase();
    if (v.includes("critical") || v.includes("unresponsive") || v.includes("very heavy")) return "var(--red)";
    if (v.includes("warning") || v.includes("slow") || v.includes("heavy") || v.includes("noticeable") || v.includes("errors")) return "var(--amber)";
    if (v.includes("no issues") || v.includes("fine") || v.includes("no errors")) return "var(--green)";
    return "var(--text-secondary)";
}

function perfRowLabel(d) {
    let name = humanizeScenarioName(d.scenario_name || "");
    const metric = (d.metric || "").toLowerCase();
    if (metric.includes("memory")) {
        name = "Memory usage — " + name.charAt(0).toLowerCase() + name.slice(1);
    }
    return name;
}

function dedupByLabel(degradations) {
    // Group by label, keep worst-case (highest peak value) per label
    const groups = {};
    for (const d of degradations) {
        const steps = d.steps || [];
        if (steps.length === 0) continue;
        const label = perfRowLabel(d);
        const peak = stepValue(steps[steps.length - 1]) || 0;
        if (!groups[label] || peak > (stepValue(groups[label].steps[groups[label].steps.length - 1]) || 0)) {
            groups[label] = d;
        }
    }
    return Object.values(groups);
}

function renderCurveChart(d, verdict) {
    const steps = d.steps || [];
    if (steps.length < 2) return "";

    const W = 300, H = 150;
    const PAD = { left: 50, right: 15, top: 12, bottom: 28 };
    const plotW = W - PAD.left - PAD.right;
    const plotH = H - PAD.top - PAD.bottom;

    const values = steps.map(s => stepValue(s));
    const labels = steps.map(s => stepLabel(s));
    const minVal = Math.min(...values);
    const maxVal = Math.max(...values);
    const range = maxVal - minVal || 1;

    function x(i) { return PAD.left + (i / (steps.length - 1)) * plotW; }
    function y(v) { return PAD.top + plotH - ((v - minVal) / range) * plotH; }

    const colour = verdictColour(verdict);
    const gridColour = "rgba(255,255,255,0.08)";
    const textColour = "var(--text-secondary)";

    let svg = `<svg width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" xmlns="http://www.w3.org/2000/svg" style="display:block">`;

    // Horizontal grid lines (3 lines)
    for (let i = 0; i <= 2; i++) {
        const val = minVal + (range * i) / 2;
        const yPos = y(val);
        svg += `<line x1="${PAD.left}" y1="${yPos}" x2="${W - PAD.right}" y2="${yPos}" stroke="${gridColour}" stroke-width="1"/>`;
        const metric = (d.metric || "").toLowerCase();
        let label;
        if (metric.includes("memory")) label = val.toFixed(1) + "MB";
        else if (metric.includes("time")) label = val.toFixed(0) + "ms";
        else label = String(Math.round(val));
        svg += `<text x="${PAD.left - 5}" y="${yPos + 3}" text-anchor="end" fill="${textColour}" font-size="9">${label}</text>`;
    }

    // Data polyline
    const points = steps.map((_, i) => `${x(i).toFixed(1)},${y(values[i]).toFixed(1)}`).join(" ");
    svg += `<polyline points="${points}" fill="none" stroke="${colour}" stroke-width="2" stroke-linejoin="round"/>`;

    // Data dots
    for (let i = 0; i < steps.length; i++) {
        svg += `<circle cx="${x(i).toFixed(1)}" cy="${y(values[i]).toFixed(1)}" r="2.5" fill="${colour}"/>`;
    }

    // Breaking point marker
    if (d.breaking_point) {
        const bpIdx = labels.indexOf(d.breaking_point);
        if (bpIdx >= 0) {
            const bpX = x(bpIdx);
            svg += `<line x1="${bpX.toFixed(1)}" y1="${PAD.top}" x2="${bpX.toFixed(1)}" y2="${PAD.top + plotH}" stroke="${colour}" stroke-width="1" stroke-dasharray="4,3" opacity="0.7"/>`;
            svg += `<circle cx="${bpX.toFixed(1)}" cy="${y(values[bpIdx]).toFixed(1)}" r="5" fill="none" stroke="${colour}" stroke-width="2"/>`;
        }
    }

    // X-axis labels: first, middle (if ≥5 steps), last
    const xLabels = [0];
    if (steps.length >= 5) xLabels.push(Math.floor(steps.length / 2));
    xLabels.push(steps.length - 1);
    for (const i of xLabels) {
        const lbl = humanizeStepLabel(labels[i]);
        // Truncate long labels
        const short = lbl.length > 15 ? lbl.slice(0, 14) + "\u2026" : lbl;
        svg += `<text x="${x(i).toFixed(1)}" y="${H - 5}" text-anchor="middle" fill="${textColour}" font-size="9">${escapeHtml(short)}</text>`;
    }

    svg += `</svg>`;
    return svg;
}

function renderDegradations(degradations, findings) {
    const rows = dedupByLabel(degradations);
    if (rows.length === 0) return "";

    let html = `<div class="degradation-section">`;
    html += `<div class="section-title">Performance under load</div>`;
    html += `<table class="perf-table">`;
    html += `<thead><tr>`;
    html += `<th>What we tested</th>`;
    html += `<th>At low load</th>`;
    html += `<th>At mid load</th>`;
    html += `<th>At peak load</th>`;
    html += `<th>Verdict</th>`;
    html += `</tr></thead><tbody>`;

    for (const d of rows) {
        const steps = d.steps || [];
        if (steps.length === 0) continue;
        const label = perfRowLabel(d);
        const low = fmtCell(steps[0], d.metric);
        let mid = "\u2014";
        if (steps.length >= 3) {
            const midIdx = Math.floor(steps.length / 2);
            mid = fmtCell(steps[midIdx], d.metric);
        }
        let high = "\u2014";
        if (steps.length >= 2) {
            high = fmtCell(steps[steps.length - 1], d.metric);
        }
        const verdict = verdictForCurve(d, findings);
        const vColour = verdictColour(verdict);

        html += `<tr>`;
        html += `<td>${escapeHtml(label)}</td>`;
        html += `<td>${escapeHtml(low)}</td>`;
        html += `<td>${escapeHtml(mid)}</td>`;
        html += `<td>${escapeHtml(high)}</td>`;
        html += `<td style="color:${vColour};font-weight:600">${escapeHtml(verdict)}</td>`;
        html += `</tr>`;

        // Inline chart row
        if (steps.length >= 2) {
            const chart = renderCurveChart(d, verdict);
            html += `<tr class="chart-row"><td colspan="5">${chart}</td></tr>`;
        }
    }

    html += `</tbody></table></div>`;
    return html;
}

function formatMetricValue(value, metric) {
    if (metric && metric.toLowerCase().includes("memory")) {
        return value.toFixed(1) + " MB";
    }
    if (metric && metric.toLowerCase().includes("time")) {
        return value.toFixed(0) + " ms";
    }
    return String(Math.round(value * 100) / 100);
}

function renderIncomplete(incomplete) {
    // Separate http_tested from other incomplete findings
    const httpTested = incomplete.filter(f => f.failure_reason === "http_tested");
    const other = incomplete.filter(f => f.failure_reason !== "http_tested");

    let html = "";

    // HTTP-tested summary (single line, never individual cards)
    if (httpTested.length > 0) {
        const framework = (httpTested[0].affected_dependencies || [])[0] || "framework";
        const n = httpTested.length;
        const verb = n === 1 ? "was" : "were";
        const plural = n === 1 ? "" : "s";
        html += `<div class="incomplete-section" style="padding:0.5rem 0;font-style:italic;color:var(--text-secondary);font-size:0.95rem">`;
        html += `${n} ${escapeHtml(framework)} scenario${plural} ${verb} tested via HTTP load testing.`;
        html += `</div>`;
    }

    // Other incomplete findings (collapsible, individual cards)
    if (other.length > 0) {
        html += `<div class="incomplete-section">`;
        html += `<button class="incomplete-toggle" onclick="toggleIncomplete(this)">`;
        html += `<span class="arrow">&#9654;</span> ${other.length} test(s) myCode could not run`;
        html += `</button>`;
        html += `<div class="incomplete-list hidden" id="incomplete-list">`;
        html += `<div class="findings-list">`;
        for (const f of other) {
            html += renderFinding(f);
        }
        html += `</div></div></div>`;
    }

    return html;
}

function toggleIncomplete(btn) {
    btn.classList.toggle("open");
    const list = $("incomplete-list");
    list.classList.toggle("hidden");
}

// ── Downloads ──

function downloadJSON() {
    const data = $("report-content").dataset.report;
    if (!data) return;
    const blob = new Blob([data], { type: "application/json" });
    downloadBlob(blob, "mycode-report.json");
}

function downloadUnderstanding() {
    const content = $("report-content");
    if (content.dataset.hasPdf && currentJobId) {
        // Download PDF from dedicated endpoint
        window.location.href = `${API}/api/report/${currentJobId}/understanding.pdf`;
        return;
    }
    // Fallback to markdown
    const md = content.dataset.understandingMd;
    if (!md) return;
    const edition = content.dataset.edition || "1";
    const blob = new Blob([md], { type: "text/markdown" });
    downloadBlob(blob, `mycode-understanding-your-results-edition-${edition}.md`);
}

function downloadBlob(blob, filename) {
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
}

