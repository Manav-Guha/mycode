/* myCode Web Interface — Application Logic */

/* global MYCODE_CONFIG */

const API = MYCODE_CONFIG.API_URL;

let currentJobId = null;
let converseTurn = 1;
let pollTimer = null;

// ── Helpers ──

function $(id) { return document.getElementById(id); }

function show(id) { $(id).classList.remove("hidden"); }
function hide(id) { $(id).classList.add("hidden"); }

function escapeHtml(str) {
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
}

async function apiPost(path, formData) {
    const res = await fetch(`${API}${path}`, { method: "POST", body: formData });
    return res.json();
}

async function apiGet(path) {
    const res = await fetch(`${API}${path}`);
    return res.json();
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
                `<div class="viability-banner fail">${escapeHtml(data.error)}</div>`;
            show("preflight-content");
            resetInput();
            return;
        }

        currentJobId = data.job_id;
        renderPreflight(data);
        show("preflight-content");

        if (data.viability && data.viability.viable) {
            beginConversation();
        }
    } catch (err) {
        hide("preflight-loading");
        $("preflight-content").innerHTML =
            `<div class="viability-banner fail">Connection failed: ${escapeHtml(err.message)}</div>`;
        show("preflight-content");
        resetInput();
    }
}

function resetInput() {
    $("go-btn").disabled = false;
    $("go-btn").textContent = "Go";
}

function renderPreflight(data) {
    const grid = $("diag-grid");
    const v = data.viability || {};

    const pct = (val) => Math.round((val || 0) * 100);
    const cls = (val, threshold) => val >= threshold ? "pass" : val >= threshold * 0.6 ? "warn" : "fail";

    grid.innerHTML = `
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
    `;

    const banner = $("viability-banner");
    if (v.viable) {
        banner.className = "viability-banner pass";
        banner.textContent = "Ready for testing";
    } else {
        banner.className = "viability-banner fail";
        banner.textContent = v.reason || "Baseline viability check failed";
    }
}

// ── Conversation ──

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
    pollProgress();
}

function pollProgress() {
    pollTimer = setInterval(async () => {
        try {
            const data = await apiGet(`/api/status/${currentJobId}`);
            updateProgress(data);

            if (data.status === "completed") {
                clearInterval(pollTimer);
                fetchReport();
            } else if (data.status === "failed") {
                clearInterval(pollTimer);
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

    hide("progress-container");
    show("report-section");
    renderReport(data.report, data.pipeline_summary);
}

function renderReport(report, summary) {
    if (!report) return;

    const content = $("report-content");
    let html = "";

    // Stats
    html += `<div class="report-stats">
        <div class="stat"><span class="stat-value">${summary.scenarios_run || 0}</span><span class="stat-label">scenarios</span></div>
        <div class="stat"><span class="stat-value" style="color:var(--green)">${summary.scenarios_passed || 0}</span><span class="stat-label">passed</span></div>
        <div class="stat"><span class="stat-value" style="color:var(--red)">${summary.scenarios_failed || 0}</span><span class="stat-label">failed</span></div>
        <div class="stat"><span class="stat-value">${summary.total_errors || 0}</span><span class="stat-label">errors</span></div>
    </div>`;

    // Summary
    if (report.summary) {
        html += `<div class="report-summary">${escapeHtml(report.summary)}</div>`;
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
    html += `<div class="download-row">
        <button class="btn btn-secondary btn-sm" onclick="downloadJSON()">Download JSON</button>
        <button class="btn btn-secondary btn-sm" onclick="downloadMarkdown()">Download Markdown</button>
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
        html += `<div class="finding-description">${escapeHtml(f.description)}</div>`;
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

function renderDegradations(degradations) {
    let html = `<div class="degradation-section">`;
    html += `<div class="section-title">Performance degradation</div>`;

    for (const d of degradations) {
        html += `<div class="degradation-card">`;
        html += `<div class="degradation-title">${escapeHtml(d.scenario_name || "").replace(/_/g, " ")}</div>`;
        html += `<div class="degradation-metric">${escapeHtml(d.metric || "")}</div>`;

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

                html += `<div class="degradation-step">`;
                html += `<span class="degradation-step-label">${escapeHtml(String(label))}</span>`;
                html += `<div class="degradation-bar-track"><div class="degradation-bar-fill ${color}" style="width:${pct}%"></div></div>`;
                html += `<span class="degradation-step-value">${displayVal}</span>`;
                html += `</div>`;
            }
            html += `</div>`;
        }

        if (d.breaking_point) {
            html += `<div style="margin-top:0.4rem;font-size:0.78rem;color:var(--red)">Breaking point: ${escapeHtml(d.breaking_point)}</div>`;
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
    let html = `<div class="incomplete-section">`;
    html += `<button class="incomplete-toggle" onclick="toggleIncomplete(this)">`;
    html += `<span class="arrow">&#9654;</span> ${incomplete.length} test(s) myCode could not run`;
    html += `</button>`;
    html += `<div class="incomplete-list hidden" id="incomplete-list">`;
    html += `<div class="findings-list">`;
    for (const f of incomplete) {
        html += renderFinding(f);
    }
    html += `</div></div></div>`;
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

function downloadMarkdown() {
    const data = $("report-content").dataset.report;
    if (!data) return;
    const report = JSON.parse(data);
    const md = reportToMarkdown(report);
    const blob = new Blob([md], { type: "text/markdown" });
    downloadBlob(blob, "mycode-report.md");
}

function downloadBlob(blob, filename) {
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
}

function reportToMarkdown(r) {
    let md = `# myCode Diagnostic Report\n\n`;
    if (r.project_name) md += `**Project:** ${r.project_name}\n\n`;
    if (r.summary) md += `## Summary\n\n${r.summary}\n\n`;

    const findings = r.findings || [];
    if (findings.length) {
        md += `## Findings\n\n`;
        for (const f of findings) {
            md += `### [${(f.severity || "info").toUpperCase()}] ${f.title}\n\n`;
            if (f.description) md += `${f.description}\n\n`;
            if (f.details) md += "```\n" + f.details + "\n```\n\n";
        }
    }

    const degradations = r.degradation_points || [];
    if (degradations.length) {
        md += `## Performance Degradation\n\n`;
        for (const d of degradations) {
            md += `**${d.scenario_name}** (${d.metric})\n\n`;
            for (const [label, value] of (d.steps || [])) {
                md += `- ${label}: ${value}\n`;
            }
            md += `\n`;
        }
    }

    md += `---\n*Generated by myCode*\n`;
    return md;
}
