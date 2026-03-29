/* myCode Web Interface — Application Logic */

/* global MYCODE_CONFIG */

const API = MYCODE_CONFIG.API_URL;

let currentJobId = null;
let converseTurn = 1;
let pollTimer = null;
let elapsedTimer = null;
let elapsedStart = null;

// Source tagging — read from URL query string (e.g. ?source=hn).
// When R shares the link with testers, she should use:
//   https://app.mycode-ai.vercel.app/?source=test_group
// For HN launch: https://app.mycode-ai.vercel.app/?source=hn
const _urlSource = new URLSearchParams(window.location.search).get("source") || "public";

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
    fd.append("source", _urlSource);
    await runPreflight(fd);
}

async function submitFile(input) {
    const file = input.files[0];
    if (!file) return;
    $("upload-label").textContent = file.name;
    const fd = new FormData();
    fd.append("file", file);
    fd.append("source", _urlSource);
    await runPreflight(fd);
}

// ── Preflight ──

async function runPreflight(formData) {
    show("columns-wrapper");
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
        showIntakeForm();
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

// ── Intake Form ──

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

function showIntakeForm() {
    show("intake-section");
    const desc = $("q-description");
    if (desc) desc.focus();
}

// Handle Enter key on URL input
document.addEventListener("DOMContentLoaded", () => {
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

function _selectedPill(name) {
    const el = document.querySelector(`input[name="${name}"]:checked`);
    return el ? el.value : "";
}

async function submitIntentForm() {
    if (!currentJobId) {
        alert("No active job. Please submit a project first.");
        return;
    }
    const btn = $("run-btn");
    btn.disabled = true;
    btn.textContent = "Starting...";

    const fd = new FormData();
    fd.append("job_id", currentJobId);
    fd.append("description", ($("q-description") || {}).value || "");
    fd.append("data_type", _selectedPill("data_type"));
    fd.append("current_users", ($("q-current-users") || {}).value || "");
    fd.append("max_users", ($("q-max-users") || {}).value || "");
    fd.append("usage_pattern", _selectedPill("usage_pattern"));
    fd.append("per_user_data", ($("q-per-user-data") || {}).value || "");
    fd.append("max_total_data", ($("q-max-total-data") || {}).value || "");
    fd.append("analysis_depth", _selectedPill("analysis_depth"));

    try {
        const data = await apiPost("/api/submit-intent", fd);
        if (data.error) {
            btn.disabled = false;
            btn.textContent = "Run Stress Tests";
            alert("Error: " + data.error);
            return;
        }
        // Immediately fetch predictions and start analysis in parallel
        fetchPredictions();
        startAnalysis();
    } catch (err) {
        btn.disabled = false;
        btn.textContent = "Run Stress Tests";
        alert("Connection failed: " + err.message);
    }
}

// ── Predictions ──

let predictionData = null;

async function fetchPredictions() {
    try {
        const data = await apiGet(`/api/predict/${currentJobId}`);
        if (data.error || !data.predictions || data.predictions.length === 0) return;
        predictionData = data;
        renderPredictions(data);
        show("prediction-section");
    } catch (err) {
        // Predictions are best-effort — don't block on failure
    }
}

function renderPredictions(data) {
    const el = $("prediction-content");
    let html = `<div class="prediction-header">Based on <strong>${data.total_similar_projects}</strong> projects with similar technology stack (${escapeHtml(data.matching_deps.slice(0, 5).join(", "))}):</div>`;
    html += '<div class="prediction-list">';
    for (const p of data.predictions) {
        const sevClass = p.severity === "critical" ? "pred-critical" : p.severity === "warning" ? "pred-warning" : "pred-info";
        html += `<div class="prediction-item ${sevClass}">`;
        html += `<span class="pred-title">${escapeHtml(p.title)}</span>`;
        html += `<span class="pred-pct">${p.probability_pct}%</span>`;
        if (p.scale_note) {
            html += `<div class="pred-note">${escapeHtml(p.scale_note)}</div>`;
        }
        html += `</div>`;
    }
    html += '</div>';
    el.innerHTML = html;
}

function annotatePredictions(findings) {
    if (!predictionData || !predictionData.predictions) return;
    const findingTitles = findings.map(f => f.title || "");
    const findingCats = findings.map(f => f.category || "");
    const el = $("prediction-content");
    if (!el) return;

    const items = el.querySelectorAll(".prediction-item");
    predictionData.predictions.forEach((p, i) => {
        if (i >= items.length) return;
        const matched = _predictionMatches(p.title, findingTitles, findingCats);
        const annotation = document.createElement("div");
        annotation.className = matched ? "pred-confirmed" : "pred-not-confirmed";
        annotation.textContent = matched
            ? "Confirmed by testing"
            : `Not observed in your project`;
        items[i].appendChild(annotation);
    });
}

function _predictionMatches(predTitle, findingTitles, findingCats) {
    const predWords = new Set(predTitle.toLowerCase().split(/\s+/).filter(w => w.length >= 3));
    predWords.delete("the"); predWords.delete("and"); predWords.delete("for");
    predWords.delete("your"); predWords.delete("with");
    for (const title of findingTitles) {
        const titleWords = new Set(title.toLowerCase().split(/\s+/).filter(w => w.length >= 3));
        let overlap = 0;
        for (const w of predWords) { if (titleWords.has(w)) overlap++; }
        if (overlap >= 2) return true;
    }
    for (const cat of findingCats) {
        const catWords = new Set(cat.toLowerCase().replace(/_/g, " ").split(/\s+/));
        for (const w of predWords) { if (catWords.has(w)) return true; }
    }
    return false;
}

// ── Analysis ──

async function startAnalysis() {
    $("run-btn").disabled = true;
    $("run-btn").textContent = "Running...";
    hide("results-placeholder");
    show("progress-section");

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
                hide("progress-section");
                show("report-section");
                $("report-content").innerHTML =
                    `<div class="finding-card critical">` +
                    `<div class="finding-header">` +
                    `<span class="severity-badge critical">error</span>` +
                    `<span class="finding-title">Analysis failed</span>` +
                    `</div>` +
                    `<div class="finding-description">${escapeHtml(data.error || "An unexpected error occurred. Please try again or use a different project.")}</div>` +
                    `</div>`;
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

    // Show completion summary in progress, then reveal report
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

    hide("progress-section");
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

    // Post-report survey — only for public source (default when no ?source= param)
    if (_urlSource === "public") {
        html += _renderSurvey();
    }

    content.innerHTML = html;

    // Store report data for downloads
    content.dataset.report = JSON.stringify(report);

    // Annotate predictions with test results
    annotatePredictions(report.findings || []);

    // Bind survey pill click handlers
    if (_urlSource === "public") {
        _bindSurveyHandlers();
    }
}

function _integrateDetails(description, details) {
    // Mirrors _integrate_details() in documents.py — merges short detail
    // fragments into the description text so they read as sentences.
    // Returns [mergedDescription, remainingDetails] where remainingDetails
    // is shown in the grey box (empty if fully integrated).
    if (!details) return [description, ""];
    if (!description) return [details, ""];
    // If details already appear in description, skip
    if (description.toLowerCase().includes(details.toLowerCase().slice(0, 20))) {
        return [description, ""];
    }
    const detailClean = details.replace(/[. ]+$/, "");
    if (detailClean.toLowerCase().startsWith("at ")) {
        // "at first iteration" → "This issue begins at first iteration."
        let desc = description.trimEnd();
        if (!desc.endsWith(".")) desc += ".";
        return [desc + " This issue begins " + detailClean + ".", ""];
    }
    // Short single-line details: integrate as sentence (matches PDF path)
    if (!details.includes("\n") && details.length < 100) {
        const desc = description.replace(/[. ]+$/, "");
        return [desc + ". " + detailClean + ".", ""];
    }
    // Multi-line or long technical details: keep in grey box
    return [description, details];
}

const _severityLabels = {
    critical: "priority",
    warning: "opportunity",
    info: "info",
};

function renderFinding(f) {
    const severity = f.severity || "info";
    const badgeLabel = _severityLabels[severity] || severity;
    let html = `<div class="finding-card ${severity}">`;
    html += `<div class="finding-header">`;
    html += `<span class="severity-badge ${severity}">${badgeLabel}</span>`;
    html += `<span class="finding-title">${escapeHtml(f.title || "")}</span>`;
    html += `</div>`;

    const [mergedDesc, remainingDetails] = _integrateDetails(
        f.description || "", f.details || "",
    );
    if (mergedDesc) {
        html += `<div class="finding-description">${highlightConsequence(mergedDesc, severity)}</div>`;
    }
    if (remainingDetails) {
        html += `<div class="finding-details">${escapeHtml(remainingDetails)}</div>`;
    }
    if (f.diagnosis) {
        html += `<div class="finding-diagnosis">${escapeHtml(f.diagnosis)}</div>`;
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

function shortStepLabel(label) {
    // Extract just the number for compact chart x-axis labels.
    const s = String(label);
    const m = s.match(/(\d[\d,]*)/);
    if (m) {
        const n = Number(m[1].replace(/,/g, ""));
        return n >= 1000 ? (n / 1000) + "k" : String(n);
    }
    return s.length > 6 ? s.slice(0, 5) + "\u2026" : s;
}

function chartAxisTitle(labels) {
    // Derive a short axis title from the step label pattern.
    const first = String(labels[0] || "");
    if (first.match(/^concurrent_/)) return "connections";
    if (first.match(/^\d+ concurrent$/)) return "connections";
    if (first.match(/^api_concurrency_/)) return "API calls";
    if (first.match(/^wsgi_concurrent_/)) return "requests";
    if (first.match(/^data_size_/)) return "items";
    if (first.match(/^compute_/)) return "items";
    if (first.match(/^rerun_rows_/)) return "rows";
    if (first.match(/^batch_/)) return "iteration";
    if (first.match(/^io_size_/)) return "data size";
    if (first.match(/^gil_threads_/)) return "threads";
    if (first.match(/^async_load_/)) return "promises";
    if (first.match(/^state_cycles_/)) return "cycles";
    if (first.match(/^sqlite_writers_/)) return "writers";
    return "";
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
     * Mirrors _finding_severity_for_dp in documents.py exactly.
     */
    if (!findings || findings.length === 0) return null;

    const metric = (d.metric || "").toLowerCase();
    const isTime = metric.includes("time");
    const isMemory = metric.includes("memory");
    const dpName = (d.scenario_name || "").toLowerCase();

    const RESPONSE_KW = ["response time", "degradation", "latency"];
    const MEMORY_KW = ["memory"];

    // Two tracking variables — mirrors Python's best / fallback_best
    let best = null;
    let fallbackBest = null;

    for (const f of findings) {
        if (f.severity !== "critical" && f.severity !== "warning") continue;

        let matched = false;
        const title = (f.title || "").toLowerCase();

        // Rule 1: HTTP category ↔ http_ scenario prefix + metric filter
        if (f.category === "http_load_testing" && dpName.startsWith("http_")) {
            if (isTime || isMemory) {
                const kws = isTime ? RESPONSE_KW : MEMORY_KW;
                if (kws.some(kw => title.includes(kw))) {
                    matched = true;
                } else {
                    // Category matches but metric doesn't — track as fallback
                    if (f.severity === "critical") {
                        fallbackBest = "critical";
                    } else if (fallbackBest === null) {
                        fallbackBest = "warning";
                    }
                    continue;
                }
            } else {
                matched = true;
            }
        }
        // Rule 2: Memory category ↔ memory metric
        else if (f.category === "memory_profiling" && isMemory) {
            matched = true;
        }
        // Rule 3: Category token in scenario name
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

        // Rule 4: Dependency overlap as tiebreaker — with metric awareness
        if (!matched && f.affected_dependencies) {
            for (const dep of f.affected_dependencies) {
                if (dpName.includes(dep.toLowerCase().replace(/-/g, "_"))) {
                    // Memory finding should not match time curve, and vice versa
                    if (isTime && MEMORY_KW.some(kw => title.includes(kw))) continue;
                    if (isMemory && RESPONSE_KW.some(kw => title.includes(kw))) continue;
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

    // Primary metric-matched results take precedence over fallback
    if (best !== null) return best;
    return fallbackBest;
}

function verdictForCurve(d, findings) {
    const steps = d.steps || [];
    if (steps.length === 0) return "";

    // Flat-curve override: <15% growth with ≥2 steps → Stable regardless of findings
    const firstVal = stepValue(steps[0]);
    const lastVal = stepValue(steps[steps.length - 1]);
    if (steps.length >= 2 && firstVal > 0 && lastVal / firstVal < 1.15) return "Stable";

    // Finding-based verdict takes precedence
    const findingSev = _findingSeverityForDp(d, findings);
    if (findingSev === "critical") return "Critical \u2014 see above";
    if (findingSev === "warning") return "Warning \u2014 see above";

    // Threshold-based fallback
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
    if (v.includes("no issues") || v.includes("fine") || v.includes("no errors") || v.includes("stable")) return "var(--green)";
    return "var(--text-secondary)";
}

function perfRowLabel(d) {
    // Use pre-computed label from JSON (matches PDF exactly)
    if (d.display_label) return d.display_label;
    // Fallback for cached reports without display_label
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

let _chartIdCounter = 0;

function _showChartTip(chartId, cx, cy, text, svgW) {
    let tip = document.getElementById("chart-tooltip-" + chartId);
    if (!tip) return;
    tip.textContent = text;
    tip.style.display = "block";
    // Position above the point, clamped within SVG bounds
    const tipW = tip.offsetWidth || 80;
    let left = cx - tipW / 2;
    if (left < 2) left = 2;
    if (left + tipW > svgW - 2) left = svgW - tipW - 2;
    let top = cy - 28;
    if (top < 2) top = cy + 12; // flip below if too close to top
    tip.style.left = left + "px";
    tip.style.top = top + "px";
}

function _hideChartTip(chartId) {
    let tip = document.getElementById("chart-tooltip-" + chartId);
    if (tip) tip.style.display = "none";
}

function renderCurveChart(d, verdict) {
    const steps = d.steps || [];
    if (steps.length < 2) return "";

    const chartId = _chartIdCounter++;
    const W = 300, H = 150;
    const PAD = { left: 50, right: 15, top: 12, bottom: 28 };
    const plotW = W - PAD.left - PAD.right;
    const plotH = H - PAD.top - PAD.bottom;

    const values = steps.map(s => stepValue(s));
    const labels = steps.map(s => stepLabel(s));
    const minVal = Math.min(...values);
    const maxVal = Math.max(...values);
    const range = maxVal - minVal || 1;

    const metric = (d.metric || "").toLowerCase();
    function x(i) { return PAD.left + (i / (steps.length - 1)) * plotW; }
    function y(v) { return PAD.top + plotH - ((v - minVal) / range) * plotH; }

    function fmtVal(v) {
        if (metric.includes("memory")) return v.toFixed(1) + " MB";
        if (metric.includes("time")) return v.toFixed(0) + "ms";
        return String(Math.round(v * 100) / 100);
    }

    const bpIdx = d.breaking_point ? labels.indexOf(d.breaking_point) : -1;
    const colour = verdictColour(verdict);
    const gridColour = "rgba(255,255,255,0.08)";
    const textColour = "var(--text-secondary)";

    // Wrapper div with relative positioning for tooltip
    let html = `<div class="chart-wrapper" style="position:relative;display:inline-block">`;
    html += `<div class="chart-tooltip" id="chart-tooltip-${chartId}"></div>`;
    html += `<svg width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" xmlns="http://www.w3.org/2000/svg" style="display:block">`;

    // Horizontal grid lines (3 lines)
    for (let i = 0; i <= 2; i++) {
        const val = minVal + (range * i) / 2;
        const yPos = y(val);
        html += `<line x1="${PAD.left}" y1="${yPos}" x2="${W - PAD.right}" y2="${yPos}" stroke="${gridColour}" stroke-width="1"/>`;
        let glabel;
        if (metric.includes("memory")) glabel = val.toFixed(1) + "MB";
        else if (metric.includes("time")) glabel = val.toFixed(0) + "ms";
        else glabel = String(Math.round(val));
        html += `<text x="${PAD.left - 5}" y="${yPos + 3}" text-anchor="end" fill="${textColour}" font-size="9">${glabel}</text>`;
    }

    // Data polyline
    const points = steps.map((_, i) => `${x(i).toFixed(1)},${y(values[i]).toFixed(1)}`).join(" ");
    html += `<polyline points="${points}" fill="none" stroke="${colour}" stroke-width="2" stroke-linejoin="round"/>`;

    // Data dots (visible)
    for (let i = 0; i < steps.length; i++) {
        html += `<circle cx="${x(i).toFixed(1)}" cy="${y(values[i]).toFixed(1)}" r="2.5" fill="${colour}"/>`;
    }

    // Breaking point marker
    if (bpIdx >= 0) {
        const bpX = x(bpIdx);
        html += `<line x1="${bpX.toFixed(1)}" y1="${PAD.top}" x2="${bpX.toFixed(1)}" y2="${PAD.top + plotH}" stroke="${colour}" stroke-width="1" stroke-dasharray="4,3" opacity="0.7"/>`;
        html += `<circle cx="${bpX.toFixed(1)}" cy="${y(values[bpIdx]).toFixed(1)}" r="5" fill="none" stroke="${colour}" stroke-width="2"/>`;
    }

    // Intent markers — user_baseline (current) and user_ceiling (target)
    if (d.user_baseline || d.user_ceiling) {
        // Find x position by matching step labels to user values
        const stepNums = labels.map(l => {
            const m = l.match(/(\d+)/);
            return m ? parseInt(m[1]) : null;
        });
        function intentX(val) {
            // Interpolate position between steps
            for (let i = 0; i < stepNums.length - 1; i++) {
                if (stepNums[i] !== null && stepNums[i + 1] !== null) {
                    if (val >= stepNums[i] && val <= stepNums[i + 1]) {
                        const frac = (val - stepNums[i]) / (stepNums[i + 1] - stepNums[i]);
                        return x(i) + frac * (x(i + 1) - x(i));
                    }
                }
            }
            // Clamp to range
            if (stepNums[0] !== null && val <= stepNums[0]) return x(0);
            if (stepNums[stepNums.length - 1] !== null && val >= stepNums[stepNums.length - 1]) return x(stepNums.length - 1);
            return null;
        }
        if (d.user_baseline) {
            const bx = intentX(d.user_baseline);
            if (bx !== null) {
                html += `<line x1="${bx.toFixed(1)}" y1="${PAD.top}" x2="${bx.toFixed(1)}" y2="${PAD.top + plotH}" stroke="var(--green)" stroke-width="1" stroke-dasharray="4,2" opacity="0.7"/>`;
                html += `<text x="${bx.toFixed(1)}" y="${PAD.top - 2}" text-anchor="middle" fill="var(--green)" font-size="8">Current</text>`;
            }
        }
        if (d.user_ceiling) {
            const cx = intentX(d.user_ceiling);
            if (cx !== null) {
                html += `<line x1="${cx.toFixed(1)}" y1="${PAD.top}" x2="${cx.toFixed(1)}" y2="${PAD.top + plotH}" stroke="var(--orange)" stroke-width="1" stroke-dasharray="4,2" opacity="0.7"/>`;
                html += `<text x="${cx.toFixed(1)}" y="${PAD.top - 2}" text-anchor="middle" fill="var(--orange)" font-size="8">Target</text>`;
            }
        }
    }

    // Invisible hover hit areas + tooltip triggers
    for (let i = 0; i < steps.length; i++) {
        const cx = x(i).toFixed(1);
        const cy = y(values[i]).toFixed(1);
        const xLabel = humanizeStepLabel(labels[i]);
        const yLabel = fmtVal(values[i]);
        let tipText = xLabel + ": " + yLabel;
        if (i === bpIdx) tipText += " \u25C6 Breaking point";
        const escaped = tipText.replace(/'/g, "\\'").replace(/"/g, "&quot;");
        html += `<circle cx="${cx}" cy="${cy}" r="15" fill="transparent" style="cursor:pointer" `
            + `onmouseover="_showChartTip(${chartId},${cx},${cy},'${escaped}',${W})" `
            + `onmouseout="_hideChartTip(${chartId})"/>`;
    }

    // X-axis labels: first, middle (if ≥5 steps), last — compact numbers
    const xLabels = [0];
    if (steps.length >= 5) xLabels.push(Math.floor(steps.length / 2));
    xLabels.push(steps.length - 1);
    for (const i of xLabels) {
        const short = shortStepLabel(labels[i]);
        html += `<text x="${x(i).toFixed(1)}" y="${H - 5}" text-anchor="middle" fill="${textColour}" font-size="9">${escapeHtml(short)}</text>`;
    }

    // X-axis title (e.g. "connections", "items")
    const axisTitle = chartAxisTitle(labels);
    if (axisTitle) {
        html += `<text x="${W - PAD.right}" y="${H - 5}" text-anchor="end" fill="${textColour}" font-size="8" font-style="italic">${escapeHtml(axisTitle)}</text>`;
    }

    html += `</svg></div>`;
    return html;
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
    _logDownload("json");
}

async function downloadUnderstanding() {
    const content = $("report-content");
    if (content.dataset.hasPdf && currentJobId) {
        // Fetch PDF via API (same base URL as all other calls)
        try {
            const res = await fetch(`${API}/api/report/${currentJobId}/understanding.pdf`);
            if (res.ok) {
                const blob = await res.blob();
                downloadBlob(blob, "mycode-understanding-your-results.pdf");
                _logDownload("pdf");
                return;
            }
        } catch (err) {
            // Fall through to markdown
        }
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

function _logDownload(type) {
    // Fire-and-forget analytics ping — don't block the download
    if (!currentJobId) return;
    const fd = new FormData();
    fd.append("type", type);
    fetch(`${API}/api/report/${currentJobId}/download-log`, { method: "POST", body: fd }).catch(() => {});
}

// ── Post-Report Survey ──

const _surveyAnswers = { q1: null, q2: null, q3: null };
let _surveySubmitted = false;

function _renderSurvey() {
    return `<div class="survey-section" id="survey-section">
        <div class="survey-title">Quick feedback (optional)</div>
        <div class="survey-q">
            <span class="survey-label">Did you expect your code to have these issues?</span>
            <div class="survey-pills" data-q="q1">
                <button class="pill" data-val="yes">Yes</button>
                <button class="pill" data-val="no">No</button>
                <button class="pill" data-val="some">Some of them</button>
            </div>
        </div>
        <div class="survey-q">
            <span class="survey-label">Was this report useful?</span>
            <div class="survey-pills" data-q="q2">
                <button class="pill" data-val="yes">Yes</button>
                <button class="pill" data-val="somewhat">Somewhat</button>
                <button class="pill" data-val="no">No</button>
            </div>
        </div>
        <div class="survey-q">
            <span class="survey-label">Will you fix these issues and retest?</span>
            <div class="survey-pills" data-q="q3">
                <button class="pill" data-val="yes">Yes</button>
                <button class="pill" data-val="maybe">Maybe</button>
                <button class="pill" data-val="no">No</button>
            </div>
        </div>
        <div class="survey-thanks hidden" id="survey-thanks">Thanks for your feedback!</div>
    </div>`;
}

function _bindSurveyHandlers() {
    document.querySelectorAll(".survey-pills .pill").forEach(btn => {
        btn.addEventListener("click", () => {
            if (_surveySubmitted) return;
            const group = btn.closest(".survey-pills");
            const q = group.dataset.q;
            const val = btn.dataset.val;
            // Deselect siblings, select this one
            group.querySelectorAll(".pill").forEach(b => b.classList.remove("pill-selected"));
            btn.classList.add("pill-selected");
            _surveyAnswers[q] = val;
            // Auto-submit when all answered
            if (_surveyAnswers.q1 && _surveyAnswers.q2 && _surveyAnswers.q3) {
                _submitSurvey();
            }
        });
    });
}

function _submitSurvey() {
    if (_surveySubmitted || !currentJobId) return;
    _surveySubmitted = true;
    // Show thanks immediately — don't wait for network
    const questions = document.querySelectorAll("#survey-section .survey-q");
    questions.forEach(q => q.classList.add("hidden"));
    const thanks = document.getElementById("survey-thanks");
    if (thanks) thanks.classList.remove("hidden");
    // Fire-and-forget POST
    fetch(`${API}/api/report/${currentJobId}/survey`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(_surveyAnswers),
    }).catch(() => {});
}

