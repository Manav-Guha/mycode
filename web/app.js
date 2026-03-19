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
    if (c.timeout_per_scenario) parts.push(`${c.timeout_per_scenario}s per test`);
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
    content.dataset.fixesMd = fixesMd || "";
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

    // Degradation curves
    const degradations = report.degradation_points || [];
    if (degradations.length > 0) {
        html += renderDegradations(degradations);
    }

    // Incomplete tests
    const incomplete = report.incomplete_tests || [];
    if (incomplete.length > 0) {
        html += renderIncomplete(incomplete);
    }

    // Download buttons
    const pdfLabel = hasPdf ? "PDF" : "Markdown";
    html += `<div class="download-row">
        <button class="btn btn-secondary btn-sm" onclick="downloadJSON()">Download JSON</button>
        <button class="btn btn-secondary btn-sm" onclick="downloadUnderstanding()">Understanding Your Results (${pdfLabel})</button>
        <button class="btn btn-secondary btn-sm" onclick="downloadFixes()">Recommended Fixes (${pdfLabel})</button>
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

function humanizeScenarioName(name) {
    if (!name) return name;
    const lower = name.toLowerCase();
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
    const s = name.replace(/_/g, " ");
    return s.charAt(0).toUpperCase() + s.slice(1);
}

function renderDegradations(degradations) {
    let html = `<div class="degradation-section">`;
    html += `<div class="section-title">Performance degradation</div>`;

    for (const d of degradations) {
        html += `<div class="degradation-card">`;
        let title = humanizeScenarioName(d.scenario_name || "");
        const metricLabel = humanizeMetricLabel(d.metric || "");
        if (metricLabel) title = metricLabel + " — " + title;
        if (d.group_count > 1) title += ` (and ${d.group_count - 1} similar)`;
        html += `<div class="degradation-title">${escapeHtml(title)}</div>`;

        const steps = d.steps || [];
        if (steps.length > 0) {
            // Find max value for bar scaling
            const maxVal = Math.max(...steps.map(s => Math.abs(s[1] || 0)), 1);

            html += `<div class="degradation-steps">`;
            for (const [label, value] of steps) {
                const pct = Math.min(100, Math.round((Math.abs(value) / maxVal) * 100));
                const ratio = Math.abs(value) / maxVal;
                const color = ratio < 0.4 ? "green" : ratio < 0.7 ? "amber" : "red";
                const displayVal = formatMetricValue(value, d.metric);
                const stepLabel = humanizeStepLabel(label);

                html += `<div class="degradation-step">`;
                html += `<span class="degradation-step-label">${escapeHtml(stepLabel)}</span>`;
                html += `<div class="degradation-bar-track"><div class="degradation-bar-fill ${color}" style="width:${pct}%"></div></div>`;
                html += `<span class="degradation-step-value">${displayVal}</span>`;
                html += `</div>`;
            }
            html += `</div>`;
        }

        if (d.breaking_point) {
            const bpLabel = humanizeStepLabel(d.breaking_point);
            // Find value at breaking point to add metric context
            let bpText = "at " + bpLabel;
            const bpStep = (d.steps || []).find(s => s[0] === d.breaking_point || s.label === d.breaking_point);
            if (bpStep) {
                const bpVal = bpStep[1] !== undefined ? bpStep[1] : bpStep.value;
                const metric = (d.metric || "").toLowerCase();
                if (bpVal !== undefined) {
                    if (metric.includes("time")) bpText = "response time exceeds " + formatMetricValue(bpVal, d.metric) + " at " + bpLabel;
                    else if (metric.includes("memory")) bpText = "memory exceeds " + formatMetricValue(bpVal, d.metric) + " at " + bpLabel;
                    else if (metric.includes("error")) bpText = "errors begin at " + bpLabel;
                }
            }
            html += `<div style="margin-top:0.4rem;font-size:0.78rem;color:var(--red)">Breaking point: ${escapeHtml(bpText)}</div>`;
        }

        html += `</div>`;
    }

    html += `</div>`;
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

function downloadFixes() {
    const content = $("report-content");
    if (content.dataset.hasPdf && currentJobId) {
        // Download PDF from dedicated endpoint
        window.location.href = `${API}/api/report/${currentJobId}/fixes.pdf`;
        return;
    }
    // Fallback to markdown
    const md = content.dataset.fixesMd;
    if (!md) return;
    const edition = content.dataset.edition || "1";
    const blob = new Blob([md], { type: "text/markdown" });
    downloadBlob(blob, `mycode-recommended-fixes-edition-${edition}.md`);
}

function downloadBlob(blob, filename) {
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
}

