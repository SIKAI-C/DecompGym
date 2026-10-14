"use strict";

const $ = (id) => document.getElementById(id);
let latest = null;
let busy = false;
let filtersReady = false;
let detailRequest = 0;
const labels = {running: "Running", recovering: "Recovering", draining_for_recovery: "Saving before recovery", complete: "Complete", queued: "Queued", in_progress: "In progress", awaiting_setup: "Runner pending", cooling: "Cooling", stopped: "Stopped", infrastructure_stopped: "Needs attention", resource_stopped: "Resource stop", supervisor_offline: "Supervisor offline", heartbeat_stale: "Heartbeat stale", gpt_complete_waiting_for_agy: "GPT complete · AGY pending", not_started: "Not started"};
labels.draining_for_pause = "Finishing current cases";
labels.paused = "Paused";
labels.later = "Later run";
labels.repair_required = "Needs repair";
const outcomeLabels = {ok: "Passed", timeout: "Timed out", agent_failed: "Generation failed", exec_failed: "Execution failed", verify_failed: "Check failed", feasible_not_optimal: "Gap above limit", objective_not_available: "No objective", reference_unavailable: "No reference"};
const num = (value, digits = 6) => value === null || value === undefined ? "—" : new Intl.NumberFormat(undefined, {maximumFractionDigits: digits}).format(value);
const pct = (value, digits = 1) => value === null || value === undefined ? "—" : `${num(value * 100, digits)}%`;
const shortNum = (value) => new Intl.NumberFormat(undefined, {notation: "compact", maximumFractionDigits: 1}).format(value || 0);
function duration(value) {
  if (value === null || value === undefined) return "—";
  if (value < 60) return `${num(value, value < 1 ? 2 : 0)}s`;
  if (value < 3600) return `${Math.floor(value / 60)}m ${Math.floor(value % 60)}s`;
  return `${Math.floor(value / 3600)}h ${Math.floor((value % 3600) / 60)}m`;
}
function localTime(value) {
  return value ? new Date(value).toLocaleTimeString([], {hour: "2-digit", minute: "2-digit", second: "2-digit"}) : "—";
}
function node(tag, text = null, className = "") {
  const element = document.createElement(tag);
  if (text !== null) element.textContent = String(text);
  if (className) element.className = className;
  return element;
}
function badge(text, state) {
  const color = ["running", "complete", "success", "queued", "historical"].includes(state) ? state : ["awaiting_setup", "cooling", "in_progress"].includes(state) ? "warning" : ["stopped", "infrastructure_stopped", "resource_stopped", "supervisor_offline", "heartbeat_stale", "danger"].includes(state) ? "danger" : "neutral";
  return node("span", text, `badge ${color}`);
}
function cell(text, className = "") { return node("td", text, className); }
function progressCell(completed, expected, suffix = "") {
  const td = cell(null, "model-progress");
  const label = node("div", null, "progress-label");
  label.append(node("span", num(completed)), node("small", ` / ${expected}${suffix}`));
  const bar = node("progress"); bar.max = expected || 1; bar.value = completed;
  bar.setAttribute("aria-label", `${completed} of ${expected} complete`);
  td.append(label, bar);
  return td;
}
function showModel(model) {
  $("filter-model").value = model;
  renderResults();
  document.querySelector(".results-panel").scrollIntoView({behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth", block: "start"});
}

function renderModels(data) {
  const fragment = document.createDocumentFragment();
  const history = data.historical;
  if (history) {
    const tr = node("tr", null, "historical-row");
    const first = cell();
    const button = node("button", history.name, "row-link model-name"); button.type = "button"; button.onclick = showHistorical;
    first.append(button, node("div", "Codex · Historical default effort", "model-setting"));
    const state = cell(); state.append(badge("Reused", "historical"));
    const rates = history.cohorts.map((cohort) => pct(cohort.rate)).join(" / ") || "—";
    const rateCell = cell(rates, "numeric"); rateCell.title = "First-attempt pass rates: historical plain cohort / reflection cohort";
    tr.append(first, state, progressCell(history.instances, history.instances, " · 2 cohorts"), cell("0", "numeric"), cell("See cohorts", "numeric muted"), rateCell);
    fragment.append(tr);
  }
  for (const model of data.models) {
    const tr = node("tr");
    const first = cell();
    const button = node("button", model.name, "row-link model-name"); button.type = "button"; button.onclick = () => showModel(model.id);
    const loginNote = model.backend === "AGY" && model.state === "awaiting_setup" ? " · Login verified" : "";
    first.append(button, node("div", `${model.backend} · ${model.effort}${loginNote}`, "model-setting"));
    const status = cell(); status.append(badge(labels[model.state] || model.state, model.state));
    if (model.queue_note) status.append(node("div", model.queue_note, "model-setting"));
    const terminalFailures = Object.values(model.policy_terminal_failures || {}).reduce((sum, value) => sum + value, 0);
    if (terminalFailures) status.append(node("div", `${terminalFailures} policy ended as failure`, "model-setting"));
    if (model.state === "awaiting_setup") status.title = "Google sign-in and the model catalog were verified. Connecting AGY to benchmark execution and isolation is still pending.";
    tr.append(first, status, progressCell(model.completed_pairs ?? model.completed, model.expected), cell(num(model.remaining_pairs ?? model.remaining), "numeric"), cell(model.completed ? `${model.passed} / ${model.completed}` : "—", "numeric"), cell(pct(model.pass_rate), "numeric"));
    fragment.append(tr);
  }
  $("models").replaceChildren(fragment);
  $("model-count").textContent = `${data.models.length + (history ? 1 : 0)} configurations`;
  $("model-note").textContent = "Progress counts completed plain + reflection pairs. Pass rate uses the shared first attempt." + (history ? " Historical Luna remains separate." : "") + (data.automatic_continuation?.armed ? " Gemini starts automatically after Claude." : data.execution_scope && data.execution_scope.planned < data.totals.planned ? " Later-run models are outside the current execution scope." : "");
  if (data.reporting_exclusions?.length) $("model-note").textContent += " Only included models are shown and counted.";
}

function renderWorkers(data) {
  const fragment = document.createDocumentFragment();
  for (const worker of data.active_workers) {
    const card = node("article", null, "worker-card");
    const top = node("div", null, "worker-top");
    top.append(node("strong", worker.worker.replace(/experiment_(\d+)/, (_, value) => `Worker ${Number(value) + 1}`)), node("span", duration(worker.elapsed_seconds)));
    card.append(top, node("h3", worker.model_name), node("div", worker.instance, "worker-instance"), node("div", worker.supervisor_active ? worker.phase : `Last checkpoint: ${worker.phase}`, "worker-phase"));
    fragment.append(card);
  }
  for (let index = 0; index < data.protocol.workers; index++) {
    if (data.active_workers.some((worker) => worker.worker === `experiment_${index}`)) continue;
    const card = node("article", null, "worker-card idle");
    card.append(node("div", `WORKER ${index + 1}`, "worker-top"), node("p", data.state === "running" ? "Ready for the next case" : "No active case"));
    fragment.append(card);
  }
  $("workers").replaceChildren(fragment);
  $("worker-live").className = `live-dot${data.totals.active ? " live" : ""}`;
  $("worker-note").textContent = `${data.protocol.workers} workers · up to ${data.protocol.codex_concurrency} simultaneous Codex calls. Case elapsed time includes earlier rounds and waits.`;
}

function renderHealth(data) {
  const monitor = data.monitor;
  const paused = data.state === "paused" || monitor.state === "paused";
  const cpu = monitor.docker_and_controller_cpu_percent;
  const memory = monitor.system_free_memory_percent;
  $("cpu").textContent = cpu == null ? "—" : `${num(cpu, 1)}%`;
  $("cpu-bar").value = cpu || 0;
  $("memory").textContent = memory == null ? "—" : `${num(memory, 0)}%`;
  $("memory-bar").value = memory || 0;
  const warnings = monitor.thermal_warnings || [];
  $("thermal").textContent = data.heartbeat_age_seconds === null ? "Unknown" : warnings.length ? warnings.join(", ") : "None reported";
  $("wls-limit").textContent = "Local academic license";
  $("experiment-memory").textContent = monitor.rss_mib == null ? "—" : `${num(monitor.rss_mib, 0)} MiB / 2,048 MiB`;
  $("resource-age").textContent = monitor.sampled_at ? `Sampled ${localTime(monitor.sampled_at)} · ${paused ? "monitoring paused" : "every ~5s"}` : "Waiting for a monitor sample";
  const stale = data.heartbeat_age_seconds === null || data.heartbeat_age_seconds > 90;
  const state = paused ? "paused" : stale ? "warning" : monitor.state === "running" ? "success" : "warning";
  const value = paused ? "Paused" : stale ? "Stale / unavailable" : monitor.state === "running" ? "Healthy" : labels[monitor.state] || monitor.state;
  $("health-badge").replaceWith(Object.assign(badge(value, state), {id: "health-badge"}));
  $("resource-note").textContent = paused ? "The experiment is paused. Resource values are the last saved sample; execution resumes only after an explicit request." : monitor.reason || "One solver thread per worker; 250% CPU and 2 GiB total experiment guard. Heat or resource pressure stops computation, then the watchdog waits for recovery.";
}

function initializeFilters(data) {
  if (filtersReady) return;
  for (const model of data.models) {
    const option = node("option", model.name); option.value = model.id; $("filter-model").append(option);
  }
  for (const [key, label] of Object.entries(data.families)) {
    const option = node("option", `${key.toUpperCase()} · ${label}`); option.value = key; $("filter-problem").append(option);
  }
  filtersReady = true;
}

function renderResults() {
  if (!latest) return;
  const model = $("filter-model").value, problem = $("filter-problem").value, size = $("filter-size").value, outcome = $("filter-outcome").value, query = $("filter-search").value.toLowerCase().trim();
  const rows = latest.results.filter((row) => (!model || row.model === model) && (!problem || row.problem === problem) && (!size || row.size === size) && (!outcome || row.passed === (outcome === "pass")) && (!query || `${row.instance} ${row.model_name}`.toLowerCase().includes(query)));
  const fragment = document.createDocumentFragment();
  for (const row of rows) {
    const tr = node("tr");
    const first = cell();
    const button = node("button", row.instance, "row-link instance-link"); button.type = "button"; button.onclick = () => showCase(row.model, row.instance); first.append(button);
    const outcomeCell = cell(); outcomeCell.append(badge(outcomeLabels[row.status] || row.status, row.passed ? "success" : "danger"));
    const time = cell(localTime(row.finished_at), "numeric muted"); time.title = new Date(row.finished_at).toLocaleString();
    tr.append(first, cell(row.model_name, "model-cell"), outcomeCell, cell(num(row.objective), "numeric"), cell(num(row.reference), "numeric muted"), cell(pct(row.gap, 3), `numeric ${row.gap === null ? "muted" : row.gap <= .01 ? "gap-good" : "gap-bad"}`), cell(duration(row.solver_seconds), "numeric"), time);
    fragment.append(tr);
  }
  if (!rows.length) { const tr = node("tr"); const td = cell(latest.results.length ? "No completed cases match these filters." : "Results will appear as each experiment finishes.", "empty"); td.colSpan = 8; tr.append(td); fragment.append(tr); }
  $("results").replaceChildren(fragment);
  $("result-count").textContent = num(latest.results.length);
  $("filter-count").textContent = `Showing ${num(rows.length)} of ${num(latest.results.length)} completed results`;
}

function renderRunDetails(data) {
  const fragment = document.createDocumentFragment();
  function group(title, paragraphs) { const section = node("section", null, "detail-group"); section.append(node("h3", title)); for (const text of paragraphs) section.append(node("p", text)); return section; }
  const elapsed = (Date.now() - new Date(data.started_at).getTime()) / 1000;
  fragment.append(group("Protocol", [`${data.protocol.instances} fixed instances · ${data.protocol.families} problem families · up to 10 attempts per policy, stopping after success. Plain and reflection share the first attempt.`, `Judge: ${data.protocol.judge} ${data.protocol.judge_effort}. Solver deadline: ${data.protocol.solver_timeout}s.`, `Elapsed since campaign creation: ${duration(elapsed)} (includes pauses).`, `Campaign: ${data.campaign}`]));
  const usage = data.usage;
  fragment.append(group("Recorded token usage", [`Generation: ${shortNum(usage.generation.input_tokens)} input · ${shortNum(usage.generation.output_tokens)} output.`, `Judge: ${shortNum(usage.judge.input_tokens)} input · ${shortNum(usage.judge.output_tokens)} output.`, `Reflection: ${shortNum(usage.reflection?.input_tokens)} input · ${shortNum(usage.reflection?.output_tokens)} output.`, "Raw input/output fields are shown here. See the audited report for provider-specific cache and reasoning fields.", data.usage_note || "Committed calls only. In-flight and failed infrastructure calls are excluded."]));
  fragment.append(group("Scheduling & recovery", [`${data.protocol.workers} experiment workers · at most ${data.protocol.codex_concurrency} active Codex invocation for the shared login.`, "Native Gurobi uses the existing local license. Durable stages resume after infrastructure backoff.", `Resource revision ${data.protocol.resource_revision} · controller revision ${data.protocol.controller_revision}.`, `${data.protocol.recovery_generation_allowance} recorded infrastructure-recovery generation allowance. Earlier completed results are preserved.`]));
  $("run-details").replaceChildren(fragment);
}

function renderCalls(data) {
  const rows = (data.recent_calls || []).map((call) => {
    const row = node("tr"), first = cell();
    first.append(node("div", call.instance), node("div", call.model_name, "model-setting"));
    row.append(first, cell(`${call.mode} · ${call.attempt}`), cell(call.stage), cell(duration(call.elapsed_seconds), "numeric"), cell(num(call.usage.input_tokens), "numeric"), cell(num(call.usage.output_tokens), "numeric"));
    return row;
  });
  if (!rows.length) { const row = node("tr"), message = cell("No committed call metrics yet.", "empty"); message.colSpan = 6; row.append(message); rows.push(row); }
  $("recent-calls").replaceChildren(...rows);
}

function render(data) {
  latest = data;
  $("run-state").replaceWith(Object.assign(badge(labels[data.state] || data.state, data.state), {id: "run-state"}));
  $("subtitle").textContent = `${data.protocol.instances} fixed instances · ${data.protocol.families} problem families · ${data.models.length} new model configurations`;
  const automatic = Boolean(data.automatic_continuation?.armed);
  const allSelected = data.execution_scope?.planned === data.totals.planned;
  const scope = automatic ? data.totals : data.execution_scope || data.totals;
  $("progress-label").textContent = automatic || allSelected ? "Full queue · completed pairs" : data.execution_scope ? "Current run · completed pairs" : "Completed policy pairs";
  $("remaining-label").textContent = automatic || allSelected ? "Remaining in queue" : data.execution_scope ? "Remaining in this run" : "Remaining";
  $("completed").textContent = num(scope.completed); $("planned").textContent = `/ ${num(scope.planned)}`;
  $("remaining").textContent = num(scope.remaining); $("pass-rate").textContent = pct(data.totals.pass_rate);
  $("pass-caption").textContent = `${num(data.totals.passed)} passed · ${num(data.totals.failed)} not passed, on the shared first attempt`;
  $("active-count").textContent = data.totals.active; $("worker-limit").textContent = `/ ${data.protocol.workers}`;
  $("worker-caption").textContent = `1 solver thread per worker · 2 GiB total guard`;
  $("total-progress").max = scope.planned; $("total-progress").value = scope.completed;
  const codex = data.models.filter((model) => model.backend === "Codex").reduce((sum, model) => sum + (model.remaining_pairs ?? model.remaining), 0);
  const agy = data.models.filter((model) => model.backend === "AGY").reduce((sum, model) => sum + (model.remaining_pairs ?? model.remaining), 0);
  $("remaining-caption").textContent = automatic ? `${data.models.length} models · continues automatically` : allSelected ? `${data.models.length} models in this queue` : data.execution_scope ? `${data.execution_scope.models.length} selected models · pauses at scope completion` : `${num(codex)} Codex · ${num(agy)} AGY`;
  $("progress-caption").textContent = automatic ? "Gemini starts automatically after Claude." : allSelected ? "All scheduled models are included." : data.execution_scope ? `Full cohort: ${num(data.totals.completed)} / ${num(data.totals.planned)} pairs` : `${pct(data.totals.completed / data.totals.planned)} of policy pairs complete`;
  $("report-link").hidden = !data.latest_report;
  if (data.latest_report) $("report-link").title = `Saved report snapshot: ${new Date(data.latest_report.saved_at).toLocaleString()}`;
  $("alerts").replaceChildren(...data.alerts.map((item) => node("div", item.text, `alert ${item.kind}`)));
  renderModels(data); renderWorkers(data); renderHealth(data); initializeFilters(data); renderResults(); renderCalls(data); renderRunDetails(data);
  $("last-updated").textContent = `Dashboard updated ${localTime(data.generated_at)}`;
}

async function refresh() {
  if (busy) return;
  busy = true; $("refresh").disabled = true;
  const controller = new AbortController(); const timeout = setTimeout(() => controller.abort(), 10000);
  try {
    const response = await fetch("/api/status", {cache: "no-store", signal: controller.signal});
    if (!response.ok) throw new Error(`Status request failed (${response.status})`);
    const data = await response.json(); render(data);
    $("connection").className = "connection online"; $("connection").textContent = $("auto-refresh").checked ? "Live · refreshes every 5s" : "Auto-refresh paused";
  } catch (error) {
    $("connection").className = "connection offline"; $("connection").textContent = "Dashboard connection lost";
    $("alerts").replaceChildren(node("div", "Cannot reach the dashboard server. Showing the last saved view; the experiment runs independently of this page.", "alert error"));
  } finally { clearTimeout(timeout); busy = false; $("refresh").disabled = false; }
}

function summaryBox(label, value) { const box = node("div"); box.append(node("span", label), node("strong", value)); return box; }
async function showCase(model, instance) {
  const request = ++detailRequest;
  $("detail-model").textContent = latest.models.find((entry) => entry.id === model)?.name || model;
  $("detail-title").textContent = instance; $("detail-body").replaceChildren(node("div", "Loading result…", "loading"));
  if (!$("detail-dialog").open) $("detail-dialog").showModal();
  try {
    const response = await fetch(`/api/case?${new URLSearchParams({model, instance})}`);
    if (!response.ok) throw new Error("The completed result is not available yet.");
    const data = await response.json(); if (request !== detailRequest) return;
    const body = document.createDocumentFragment();
    const metrics = node("div", null, "detail-summary");
    metrics.append(summaryBox("Objective", num(data.objective)), summaryBox("Reference", num(data.reference)), summaryBox("Objective gap", pct(data.gap, 5)), summaryBox("Solver execution", duration(data.metrics.solver_seconds)), summaryBox("Generation wall time", duration(data.metrics.generation_seconds)), summaryBox("Judge wall time", duration(data.metrics.judge_seconds)));
    body.append(metrics);
    for (const [mode, policy] of Object.entries(data.policies || {})) {
      if (["user_adjudicated_terminal_failure", "terminal_reflection_delivery_failure"].includes(policy.completion_kind)) {
        body.append(node("p", `${mode === "reflection" ? "Reflection" : "Plain"} policy ended as failure because the required reflection file was not produced. ${policy.actual_attempts} actual attempt; ${policy.unexecuted_attempt_slots} remaining budget slots were not run.`, "alert warning"));
      }
    }
    const checks = node("div", null, "checks");
    checks.append(badge(data.execution_ok ? "Execution passed" : "Execution failed", data.execution_ok ? "success" : "danger"), badge(data.feasibility_ok ? "Feasibility passed" : "Feasibility not passed", data.feasibility_ok ? "success" : "danger"), badge(data.algorithm_verdict === "pass" ? "Algorithm passed" : `Algorithm: ${data.algorithm_verdict || "unavailable"}`, data.algorithm_verdict === "pass" ? "success" : "danger")); body.append(checks);
    if (data.issues.length) { body.append(node("h3", "Recorded issues")); const list = node("ul"); for (const issue of data.issues) list.append(node("li", issue)); body.append(list); }
    body.append(node("p", "Client-call durations and solver execution time are recorded separately. The worker's case elapsed time also includes earlier rounds and waits.", "muted"));
    for (const [filename, label] of [["alg_verifier.md", "Algorithm judge notes"], ["solver.py", "Generated solver code"], ["exec.log", "Execution log"]]) {
      const details = node("details", null, "artifact"); details.open = filename === "alg_verifier.md";
      details.append(node("summary", label), node("pre", data.files[filename] || "No artifact available.")); body.append(details);
    }
    $("detail-body").replaceChildren(body);
  } catch (error) { if (request === detailRequest) $("detail-body").replaceChildren(node("div", error.message, "alert error")); }
}

function showHistorical() {
  ++detailRequest;
  const history = latest.historical;
  $("detail-model").textContent = "REUSED HISTORICAL RESULTS"; $("detail-title").textContent = history.name;
  const body = document.createDocumentFragment();
  body.append(node("p", history.note));
  for (const cohort of history.cohorts) {
    body.append(node("h3", cohort.mode === "plain" ? "Without-reflection cohort · first attempt" : "With-reflection cohort · first attempt"));
    const boxes = node("div", null, "detail-summary"); boxes.append(summaryBox("Execution", `${cohort.execution} / ${cohort.instances}`), summaryBox("Verified", `${cohort.verified} / ${cohort.instances}`), summaryBox("Quality passes", `${cohort.passed} / ${cohort.instances} · ${pct(cohort.rate)}`)); body.append(boxes);
  }
  body.append(node("p", "Both values describe first attempts, before any reflection feedback. There are 81 shared instances and two saved cohorts (162 trajectories). No new Luna generation is scheduled.", "muted"));
  $("detail-body").replaceChildren(body); if (!$("detail-dialog").open) $("detail-dialog").showModal();
}

$("refresh").addEventListener("click", refresh);
$("auto-refresh").addEventListener("change", () => { if ($("auto-refresh").checked) refresh(); else $("connection").textContent = "Auto-refresh paused"; });
for (const id of ["filter-model", "filter-problem", "filter-size", "filter-outcome", "filter-search"]) $(id).addEventListener("input", renderResults);
$("clear-filters").addEventListener("click", () => { for (const id of ["filter-model", "filter-problem", "filter-size", "filter-outcome", "filter-search"]) $(id).value = ""; renderResults(); });
$("close-dialog").addEventListener("click", () => { ++detailRequest; $("detail-dialog").close(); });
$("detail-dialog").addEventListener("click", (event) => { if (event.target === $("detail-dialog")) { ++detailRequest; $("detail-dialog").close(); } });
document.addEventListener("visibilitychange", () => { if (!document.hidden && $("auto-refresh").checked) refresh(); });
setInterval(() => { if (!document.hidden && $("auto-refresh").checked) refresh(); }, 5000);
refresh();
