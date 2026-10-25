/* FraudLens AI front end - every number on screen comes from /api/* responses. */
"use strict";
const $ = (s, r = document) => r.querySelector(s);
const view = $("#view");
let charts = [];
const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const nf = new Intl.NumberFormat("en-US");
const money = v => nf.format(Math.round(v));  // PaySim does not state a currency
const compact = v => new Intl.NumberFormat("en-US", { notation: "compact", maximumFractionDigits: 2 }).format(v);
const pct = (v, d = 1) => (v * 100).toFixed(d) + "%";
let SCHEMA = null;
const C = { blue: "#2F5BEA", fraud: "#C8284B", safe: "#12806A", amber: "#D08A12", grey: "#8A97B2", purple: "#7B4FD6" };

function toast(msg, err = false) {
  const t = $("#toast"); t.textContent = msg; t.className = "show" + (err ? " err" : "");
  clearTimeout(toast.h); toast.h = setTimeout(() => (t.className = ""), 4200);
}
async function api(path, opts) {
  let res;
  try { res = await fetch(path, opts); }
  catch { throw new Error("Cannot reach the FraudLens backend. Is the server running (uvicorn backend.main:app)?"); }
  if (!res.ok) {
    let msg = res.statusText;
    try {
      const j = await res.json();
      if (Array.isArray(j.detail)) msg = j.detail.map(d => { const f = (d.loc || []).slice(1).join("."); return (f ? f + ": " : "") + String(d.msg).replace(/^Value error, /, ""); }).join("; ");
      else if (j.detail) msg = j.detail;
    } catch {}
    const e = new Error(msg); e.status = res.status; e.body = msg; throw e;
  }
  return res.json();
}
const loading = (t = "Loading data...") => (view.innerHTML = `<div class="loading"><div class="spin"></div>${t}</div>`);
const errorBox = e => (view.innerHTML = `<div class="banner err"><strong>Could not load this page.</strong> ${esc(e.message)}</div>
  <button class="btn" onclick="route()">Try again</button>`);
function mkChart(id, cfg) {
  const el = document.getElementById(id); if (!el) return;
  const ch = new Chart(el, cfg); charts.push(ch); return ch;
}
function chartDefaults() {
  const dark = matchMedia("(prefers-color-scheme: dark)").matches;
  Chart.defaults.color = dark ? "#98A5C0" : "#5E6B85";
  Chart.defaults.borderColor = dark ? "#26344F" : "#E3E8F0";
  Chart.defaults.font.family = '"Segoe UI Variable","Segoe UI",system-ui,sans-serif';
  Chart.defaults.maintainAspectRatio = false;
}
const kpi = (label, value, note = "", cls = "") =>
  `<div class="card kpi ${cls}"><div class="label">${label}</div><div class="value">${value}</div><div class="note">${note}</div></div>`;

/* ------------------------------------------------------------------ Overview */
const note = t => `<div class="banner">${t}</div>`;
async function pageOverview() {
  const [s, ins, m] = await Promise.all([api("/api/summary"), api("/api/insights"), api("/api/model-metrics")]);
  const br = s.real_base_rates;
  view.innerHTML = note("<strong>Read this first:</strong> this dataset is a <strong>balanced sample by design</strong> (50% fraud), built from three public datasets. The fraud share below is a sampling choice, not a real-world rate. Real rates in the full datasets are roughly " +
    `PaySim ${pct(br.PaySim, 2)}, ULB ${pct(br.ULB, 2)}, IEEE-CIS ${pct(br["IEEE-CIS"], 1)}.`) + `
  <div class="grid g4">
    ${kpi("Total transactions", nf.format(s.total_transactions), s.sources.map(x => `${x.Source} ${nf.format(x.count)}`).join(" · "), "primary")}
    ${kpi("Fraud cases", nf.format(s.fraud_cases), "Labelled fraud in the sample", "fraud")}
    ${kpi("Fraud share of sample", pct(s.fraud_share, 0), "Balanced by design")}
    ${kpi("Data sources", String(s.sources.length), "PaySim (mobile money), IEEE-CIS (e-commerce), ULB (card)")}
  </div>
  <h2 style="margin-top:24px">Model performance <span class="badge b-sel">${esc(s.model.name)}</span></h2>
  <p class="sub">${esc(s.model.protocol)}. Test rows are never used for training or threshold tuning.</p>
  <div class="grid g4">
    ${kpi("Precision", pct(s.model.precision), "Of flagged, share truly fraud (on balanced data)")}
    ${kpi("Recall", pct(s.model.recall), "Of real fraud, share caught")}
    ${kpi("F1-score", s.model.f1.toFixed(3), "Balance of precision and recall")}
    ${kpi("PR-AUC", s.model.pr_auc.toFixed(3), `ROC-AUC ${s.model.roc_auc.toFixed(3)}`)}
  </div>
  <div class="grid g3" style="margin-top:16px">
    <div class="card"><h2>Transactions by source</h2><p class="sub">Legitimate vs fraud</p><div class="chart"><canvas id="c1"></canvas></div></div>
    <div class="card"><h2>Fraud share by type</h2><p class="sub">Within this sample</p><div class="chart"><canvas id="c2"></canvas></div></div>
    <div class="card"><h2>Model ROC-AUC by source</h2><p class="sub">Held-out test rows only</p><div class="chart"><canvas id="c3"></canvas></div></div>
  </div>`;
  mkChart("c1", { type: "bar", data: { labels: ins.by_source.map(x => x.Source), datasets: [{ label: "Legitimate", data: ins.by_source.map(x => x.count - x.fraud), backgroundColor: C.blue }, { label: "Fraud", data: ins.by_source.map(x => x.fraud), backgroundColor: C.fraud }] }, options: { scales: { x: { stacked: true }, y: { stacked: true } } } });
  const bt = ins.by_type;
  mkChart("c2", { type: "bar", data: { labels: bt.map(x => x.Source + " · " + x.TransactionType), datasets: [{ label: "Fraud share %", data: bt.map(x => +(x.fraud_rate * 100).toFixed(1)), backgroundColor: C.fraud }] }, options: { indexAxis: "y", scales: { x: { max: 100 } }, plugins: { legend: { display: false } } } });
  const ps = m.per_source, ks = Object.keys(ps);
  mkChart("c3", { type: "bar", data: { labels: ks, datasets: [{ label: "ROC-AUC", data: ks.map(k => +ps[k].roc_auc.toFixed(3)), backgroundColor: C.safe }] }, options: { scales: { y: { min: 0.5, max: 1 } }, plugins: { legend: { display: false } } } });
}

/* ------------------------------------------------------------------ Transactions */
const TX = { page: 1, page_size: 25, sort_by: "tx_id", sort_dir: "asc", search: "", source: "ALL", type: "ALL", label: "all", model_flag: "all", split: "all", min_amount: "", max_amount: "" };
const qs = (extra = {}) => {
  const p = new URLSearchParams();
  for (const [k, v] of Object.entries({ ...TX, ...extra })) if (v !== "" && v != null) p.set(k, v);
  return p.toString();
};
const typeOptions = src => (src === "ALL" ? Object.values(SCHEMA.types_by_source).flat() : SCHEMA.types_by_source[src]);
const COLS = [["tx_id", "ID", 1], ["Source", "Source"], ["TransactionType", "Type"], ["Amount", "Amount", 1], ["Time", "Raw time", 1], ["hour_of_day", "Hour", 1], ["risk_score", "Risk score", 1], ["FraudLabel", "Actual"], ["predicted_fraud", "Model"], ["split", "Set"]];
async function pageTransactions() {
  const opts = (arr, sel) => arr.map(t => `<option ${sel === t ? "selected" : ""}>${t}</option>`).join("");
  view.innerHTML = `
  <div class="toolbar">
    <label class="f">Search ID, source or type<input id="f-search" type="search" placeholder="e.g. 1234 or TRANSFER" value="${esc(TX.search)}" maxlength="40"></label>
    <label class="f">Source<select id="f-source">${opts(["ALL", "PaySim", "IEEE-CIS", "ULB"], TX.source)}</select></label>
    <label class="f">Type<select id="f-type">${opts(["ALL", ...typeOptions(TX.source)], TX.type)}</select></label>
    <label class="f">Actual label<select id="f-label"><option value="all">All</option><option value="fraud">Fraud</option><option value="legit">Legitimate</option></select></label>
    <label class="f">Model verdict<select id="f-flag"><option value="all">All</option><option value="flagged">Flagged by model</option><option value="not_flagged">Not flagged</option></select></label>
    <label class="f">Set<select id="f-split"><option value="all">All</option><option value="test">Test (later 20%, held out)</option><option value="train">Train (earlier 80%)</option></select></label>
    <label class="f">Min amount<input id="f-min" type="number" min="0" style="width:110px" value="${esc(TX.min_amount)}"></label>
    <label class="f">Max amount<input id="f-max" type="number" min="0" style="width:110px" value="${esc(TX.max_amount)}"></label>
    <button class="btn ghost" id="f-reset">Reset filters</button>
    <a class="btn" id="f-export" href="#">Export CSV</a>
  </div>
  <p class="sub">Risk scores are honest for every row: train rows use out-of-fold predictions, test rows use the held-out model. Amounts are in each source's own units; raw time units differ by source.</p>
  <div id="tx-body"></div>`;
  $("#f-label").value = TX.label; $("#f-flag").value = TX.model_flag; $("#f-split").value = TX.split;
  const bind = (id, key) => $(id).addEventListener("change", e => { TX[key] = e.target.value; TX.page = 1; loadTx(); });
  let tm; $("#f-search").addEventListener("input", e => { clearTimeout(tm); tm = setTimeout(() => { TX.search = e.target.value; TX.page = 1; loadTx(); }, 300); });
  $("#f-source").addEventListener("change", e => { TX.source = e.target.value; TX.type = "ALL"; TX.page = 1; $("#f-type").innerHTML = opts(["ALL", ...typeOptions(TX.source)], "ALL"); loadTx(); });
  bind("#f-type", "type"); bind("#f-label", "label"); bind("#f-flag", "model_flag"); bind("#f-split", "split"); bind("#f-min", "min_amount"); bind("#f-max", "max_amount");
  $("#f-reset").onclick = () => { Object.assign(TX, { page: 1, sort_by: "tx_id", sort_dir: "asc", search: "", source: "ALL", type: "ALL", label: "all", model_flag: "all", split: "all", min_amount: "", max_amount: "" }); pageTransactions(); };
  $("#f-export").onclick = async e => { e.preventDefault(); await download("/api/export/transactions?" + qs({ page: "", page_size: "" }), "fraudlens_transactions.csv"); };
  loadTx();
}
let txToken = 0;
async function loadTx() {
  const box = $("#tx-body"); if (!box) return;
  const my = ++txToken;
  box.innerHTML = `<div class="loading"><div class="spin"></div>Loading transactions...</div>`;
  try {
    const d = await api("/api/transactions?" + qs());
    if (my !== txToken) return;
    TX.page = d.page;
    if (!d.rows.length) { box.innerHTML = `<div class="card empty"><strong>No transactions match these filters.</strong><br>Widen the amount range, clear the search box, or reset the filters.</div>`; return; }
    const arrow = k => (TX.sort_by === k ? (TX.sort_dir === "asc" ? " ▲" : " ▼") : "");
    const head = COLS.map(([k, l, n]) => `<th class="${n ? "num" : ""}" aria-sort="${TX.sort_by === k ? (TX.sort_dir === "asc" ? "ascending" : "descending") : "none"}">${k === "split" ? l : `<button data-sort="${k}">${l}${arrow(k)}</button>`}</th>`).join("");
    const rows = d.rows.map(r => `<tr>
      <td class="num">${r.tx_id}</td><td>${esc(r.Source)}</td><td>${esc(r.TransactionType)}</td><td class="num">${nf.format(r.Amount)}</td><td class="num">${nf.format(r.Time)}</td><td class="num">${r.hour_of_day}</td>
      <td class="num">${(r.risk_score * 100).toFixed(1)}%</td>
      <td><span class="badge ${r.FraudLabel ? "b-fraud" : "b-ok"}">${r.FraudLabel ? "Fraud" : "Legit"}</span></td>
      <td><span class="badge ${r.predicted_fraud ? "b-flag" : "b-ok"}">${r.predicted_fraud ? "Flagged" : "Clear"}</span></td><td>${esc(r.split)}</td></tr>`).join("");
    const from = (d.page - 1) * d.page_size + 1, to = from + d.rows.length - 1;
    let nums = []; for (let p = Math.max(1, d.page - 2); p <= Math.min(d.pages, d.page + 2); p++) nums.push(p);
    box.innerHTML = `<div class="tablewrap"><table><thead><tr>${head}</tr></thead><tbody>${rows}</tbody></table></div>
      <div class="pager"><span>Showing ${nf.format(from)}-${nf.format(to)} of ${nf.format(d.total)} transactions</span>
        <span class="btns"><label>Rows <select id="ps">${[10, 25, 50, 100].map(n => `<option ${n === d.page_size ? "selected" : ""}>${n}</option>`).join("")}</select></label>
        <button data-p="1" ${d.page === 1 ? "disabled" : ""}>First</button><button data-p="${d.page - 1}" ${d.page === 1 ? "disabled" : ""}>Prev</button>
        ${nums.map(p => `<button data-p="${p}" class="${p === d.page ? "cur" : ""}">${p}</button>`).join("")}
        <button data-p="${d.page + 1}" ${d.page === d.pages ? "disabled" : ""}>Next</button><button data-p="${d.pages}" ${d.page === d.pages ? "disabled" : ""}>Last</button></span></div>`;
    box.querySelectorAll("[data-p]").forEach(b => (b.onclick = () => { TX.page = +b.dataset.p; loadTx(); }));
    box.querySelectorAll("[data-sort]").forEach(b => (b.onclick = () => { const k = b.dataset.sort; TX.sort_dir = TX.sort_by === k && TX.sort_dir === "asc" ? "desc" : "asc"; TX.sort_by = k; TX.page = 1; loadTx(); }));
    $("#ps").onchange = e => { TX.page_size = +e.target.value; TX.page = 1; loadTx(); };
  } catch (e) { if (my === txToken) box.innerHTML = `<div class="banner err">${esc(e.message)}</div>`; }
}
async function download(url, name) {
  try {
    const res = await fetch(url);
    if (!res.ok) { let m = res.statusText; try { m = (await res.json()).detail || m; } catch {} throw new Error(typeof m === "string" ? m : JSON.stringify(m)); }
    const blob = await res.blob(); const a = document.createElement("a");
    a.href = URL.createObjectURL(blob); a.download = name; document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 2000); toast("Downloaded " + name);
  } catch (e) { toast("Export failed: " + e.message, true); }
}

/* ------------------------------------------------------------------ Detect */
async function pageDetect() {
  view.innerHTML = `
  <div class="banner">Educational model trained on a small <strong>balanced</strong> sample of public datasets (PaySim is simulated, not real UPI). Probabilities are for ranking, not real-world fraud odds.</div>
  <div class="grid g2">
   <div class="card"><h2>Transaction details</h2><p class="sub">Enter values, or load a real held-out transaction from the dataset.</p>
    <div class="examples" id="examples"><span class="sub">Loading examples...</span></div>
    <form id="pf" novalidate><div class="form-grid">
     <label class="f">Data source<select id="p-source">${["PaySim", "IEEE-CIS", "ULB"].map(o => `<option>${o}</option>`).join("")}</select><span class="err-msg" id="e-source"></span></label>
     <label class="f" id="lbl-type">Transaction type<select id="p-type"></select><span class="err-msg" id="e-type"></span></label>
     <label class="f">Amount (source's own units)<input id="p-amount" type="number" step="any" min="0" required><span class="err-msg" id="e-amount"></span></label>
     <label class="f">Hour of day (0-23)<input id="p-hour" type="number" step="1" min="0" max="23" required><span class="err-msg" id="e-hour"></span></label>
     <div class="full"><button class="btn" id="p-go" type="submit">Assess transaction</button> <button class="btn ghost" type="button" id="p-clear">Clear</button></div>
    </div></form></div>
   <div class="card" id="result"><div class="empty">Submit a transaction to see the model's fraud probability and which inputs drove it.</div></div>
  </div>`;
  const fillTypes = (src, keep) => { $("#p-type").innerHTML = SCHEMA.types_by_source[src].map(t => `<option ${t === keep ? "selected" : ""}>${t}</option>`).join(""); $("#lbl-type").firstChild.textContent = src === "IEEE-CIS" ? "Product code (ProductCD)" : "Transaction type"; };
  $("#p-source").onchange = e => fillTypes(e.target.value);
  const setForm = r => { $("#p-source").value = r.source; fillTypes(r.source, r.transaction_type); $("#p-amount").value = r.amount; $("#p-hour").value = r.hour_of_day; };
  setForm({ source: "PaySim", transaction_type: "TRANSFER", amount: 250000, hour_of_day: 3 });
  api("/api/examples").then(ex => {
    const btn = (label, r) => { const b = document.createElement("button"); b.type = "button"; b.className = "btn ghost"; b.textContent = `${label} (${r.source})`; b.onclick = () => { setForm(r); $("#pf").requestSubmit(); }; return b; };
    const box = $("#examples"); box.innerHTML = "";
    [["Real fraud (caught)", ex.fraud_caught], ["Real fraud (missed)", ex.fraud_missed], ["Real legitimate", ex.legit_clear], ["Legit but flagged", ex.false_alarm]].forEach(([l, arr]) => arr.forEach((r, i) => { if (i < 2) box.append(btn(l, r)); }));
    if (ex.fraud_caught[0]) setForm(ex.fraud_caught[0]);
  }).catch(e => ($("#examples").innerHTML = `<span class="err-msg">${esc(e.message)}</span>`));
  $("#p-clear").onclick = () => { $("#p-amount").value = ""; $("#p-hour").value = ""; ["amount", "hour", "type", "source"].forEach(k => ($("#e-" + k).textContent = "")); $("#result").innerHTML = `<div class="empty">Submit a transaction to see the model's fraud probability.</div>`; };
  $("#pf").onsubmit = async ev => {
    ev.preventDefault(); let bad = false;
    const amt = $("#p-amount").value.trim(), hr = $("#p-hour").value.trim();
    ["amount", "hour"].forEach(k => { $("#e-" + k).textContent = ""; $("#p-" + k).removeAttribute("aria-invalid"); });
    if (amt === "" || isNaN(+amt)) { $("#e-amount").textContent = "Enter a number."; bad = true; } else if (+amt < 0) { $("#e-amount").textContent = "Must be 0 or more."; bad = true; }
    if (hr === "" || !Number.isInteger(+hr) || +hr < 0 || +hr > 23) { $("#e-hour").textContent = "Whole number from 0 to 23."; bad = true; }
    if (bad) { if ($("#e-amount").textContent) $("#p-amount").setAttribute("aria-invalid", "true"); if ($("#e-hour").textContent) $("#p-hour").setAttribute("aria-invalid", "true"); return; }
    const body = { source: $("#p-source").value, transaction_type: $("#p-type").value, amount: +amt, hour_of_day: +hr };
    const go = $("#p-go"); go.disabled = true; go.textContent = "Scoring...";
    $("#result").innerHTML = `<div class="loading"><div class="spin"></div>Running the trained model...</div>`;
    try { renderPrediction(await api("/api/predict", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) })); }
    catch (e) { $("#result").innerHTML = `<div class="banner err"><strong>Prediction failed.</strong> ${esc(e.message)}</div>`; }
    go.disabled = false; go.textContent = "Assess transaction";
  };
}
function renderPrediction(r) {
  const cls = r.risk_level.toLowerCase(); const maxI = Math.max(...r.influences.map(i => Math.abs(i.influence)), 1e-9);
  $("#result").innerHTML = `
   <div class="verdict ${cls}"><div>${r.is_fraud ? "Potentially fraudulent" : "Likely legitimate"} - ${r.risk_level} risk</div>
    <div class="big">${(r.fraud_probability * 100).toFixed(1)}%</div>
    <div class="meter"><i style="width:${Math.min(100, r.fraud_probability * 100)}%"></i><b style="left:${Math.min(100, r.threshold * 100)}%" title="Decision threshold"></b></div>
    <small>Model score. Decision threshold ${(r.threshold * 100).toFixed(1)}% (black marker). Model: ${esc(r.model)}.</small></div>
   <h2>What drove this score</h2><p class="sub">Change in score when each input is swapped for a typical legitimate value of the same source (occlusion test on the real model).</p>
   <div class="bars">${r.influences.map(i => `<div class="row"><span>${esc(i.feature)}</span><div class="bar"><i style="left:${i.influence >= 0 ? 0 : 100 - Math.abs(i.influence) / maxI * 100}%;width:${Math.abs(i.influence) / maxI * 100}%;background:${i.influence >= 0 ? C.fraud : C.safe}"></i></div><span>${i.influence >= 0 ? "+" : ""}${(i.influence * 100).toFixed(1)} pts</span></div>`).join("")}</div>
   ${r.notes.map(n => `<div class="hl">${esc(n)}</div>`).join("")}<p class="sub" style="margin-top:12px">${esc(r.disclaimer)}</p>`;
}

/* ------------------------------------------------------------------ Analytics */
async function pageAnalytics() {
  const d = await api("/api/insights"); const SRC = ["PaySim", "IEEE-CIS", "ULB"]; const col = { PaySim: C.blue, "IEEE-CIS": C.amber, ULB: C.purple };
  view.innerHTML = `
  <div class="grid g3">
    ${kpi("PaySim types with zero fraud", d.paysim_zero_fraud_types.join(", "), "No fraud in these types in the sample")}
    ${kpi("PaySim TRANSFER fraud share", pct(d.paysim_transfer_fraud_rate, 0), "Within this balanced sample", "fraud")}
    ${kpi("Median amount, fraud vs legit", SRC.map(s => `${compact(d.median_amount[s].fraud)}/${compact(d.median_amount[s].legit)}`).join(" · "), "PaySim · IEEE-CIS · ULB (fraud / legit)")}
  </div>
  <div class="grid g2" style="margin-top:16px">
    <div class="card"><h2>Transactions by source and type</h2><p class="sub">Legitimate vs fraud counts</p><div class="chart"><canvas id="a1"></canvas></div></div>
    <div class="card"><h2>Fraud share by type</h2><p class="sub">Share of that type's rows labelled fraud</p><div class="chart"><canvas id="a2"></canvas></div></div>
    <div class="card"><h2>Fraud share by amount quartile</h2><p class="sub">Quartiles computed within each source (amount units differ)</p><div class="chart"><canvas id="a3"></canvas></div></div>
    <div class="card"><h2>Fraud share by hour of day</h2><p class="sub">Hours with few rows are noisy; time was matched across classes by design</p><div class="chart"><canvas id="a4"></canvas></div></div>
    <div class="card"><h2>Feature importance</h2><p class="sub">Permutation importance on test rows: drop in PR-AUC when shuffled</p><div class="chart"><canvas id="a5"></canvas></div></div>
    <div class="card"><h2>Median amount: legit vs fraud</h2><p class="sub">Log scale, each source in its own units</p><div class="chart"><canvas id="a6"></canvas></div></div>
  </div>`;
  const lab = d.by_type.map(x => x.Source + " · " + x.TransactionType);
  mkChart("a1", { type: "bar", data: { labels: lab, datasets: [{ label: "Legitimate", data: d.by_type.map(x => x.count - x.fraud), backgroundColor: C.blue }, { label: "Fraud", data: d.by_type.map(x => x.fraud), backgroundColor: C.fraud }] }, options: { scales: { x: { stacked: true }, y: { stacked: true } } } });
  mkChart("a2", { type: "bar", data: { labels: lab, datasets: [{ label: "Fraud share %", data: d.by_type.map(x => +(x.fraud_rate * 100).toFixed(1)), backgroundColor: d.by_type.map(x => col[x.Source]) }] }, options: { indexAxis: "y", scales: { x: { max: 100 } }, plugins: { legend: { display: false } } } });
  const bl = ["Q1 lowest", "Q2", "Q3", "Q4 highest"];
  mkChart("a3", { type: "bar", data: { labels: bl, datasets: SRC.map(s => ({ label: s, backgroundColor: col[s], data: bl.map(b => { const r = d.amount_bands.find(x => x.Source === s && x.band === b); return r ? +(r.fraud_rate * 100).toFixed(1) : 0; }) })) }, options: { scales: { y: { max: 100, title: { display: true, text: "Fraud share %" } } } } });
  mkChart("a4", { type: "line", data: { labels: d.hourly.map(x => x.hour_of_day + ":00"), datasets: [{ label: "Fraud share %", data: d.hourly.map(x => +(x.fraud_rate * 100).toFixed(1)), borderColor: C.amber, tension: .2 }] }, options: { plugins: { legend: { display: false } }, scales: { y: { min: 0, max: 100 } } } });
  mkChart("a5", { type: "bar", data: { labels: d.feature_importance.map(x => x.feature), datasets: [{ label: "Importance", data: d.feature_importance.map(x => +x.importance.toFixed(4)), backgroundColor: C.blue }] }, options: { indexAxis: "y", plugins: { legend: { display: false } } } });
  mkChart("a6", { type: "bar", data: { labels: SRC, datasets: [{ label: "Legitimate", data: SRC.map(s => d.median_amount[s].legit), backgroundColor: C.blue }, { label: "Fraud", data: SRC.map(s => d.median_amount[s].fraud), backgroundColor: C.fraud }] }, options: { scales: { y: { type: "logarithmic" } } } });
}

/* ------------------------------------------------------------------ Performance */
let PROTO = "time";
async function pagePerformance() {
  const m = await api("/api/model-metrics"); const P = m.protocols[PROTO]; const names = Object.keys(P.models); const colors = [C.grey, C.amber, C.safe, C.blue];
  const sel = m.selected_model;
  view.innerHTML = `
  <div class="banner ok">Selected model: <strong>${esc(sel)}</strong> (chosen under the time-based protocol). ${esc(m.selection_rule)}</div>
  <div class="tabs" id="proto"><button data-p="time" class="${PROTO === "time" ? "on" : ""}">Time-based split (primary)</button><button data-p="random" class="${PROTO === "random" ? "on" : ""}">Random split (secondary)</button></div>
  <p class="sub">${PROTO === "time" ? "Train on the earlier 80% of each source, test on the later 20% - closest to real deployment." : "Random 80/20 split. Optimistic: 24 (source, time, amount) combinations appear in both train and test."} Train ${nf.format(P.n_train)} rows, test ${nf.format(P.n_test)} rows (${P.test_fraud} fraud).</p>
  <div class="card"><h2>Model comparison (test set)</h2>
   <div class="tablewrap"><table><thead><tr><th>Model</th><th class="num">CV PR-AUC</th><th class="num">Precision</th><th class="num">Recall</th><th class="num">F1</th><th class="num">PR-AUC</th><th class="num">ROC-AUC</th><th class="num">Accuracy</th><th class="num">Threshold</th></tr></thead><tbody>
   ${names.map(n => { const r = P.models[n]; return `<tr><td>${esc(n)} ${n === sel && PROTO === "time" ? '<span class="badge b-sel">selected</span>' : ""}</td><td class="num">${r.cv.pr_auc.toFixed(3)}</td><td class="num">${r.precision.toFixed(3)}</td><td class="num">${r.recall.toFixed(3)}</td><td class="num">${r.f1.toFixed(3)}</td><td class="num">${r.pr_auc.toFixed(3)}</td><td class="num">${r.roc_auc.toFixed(3)}</td><td class="num">${(r.accuracy * 100).toFixed(1)}%</td><td class="num">${r.threshold.toFixed(3)}</td></tr>`; }).join("")}
   </tbody></table></div></div>
  <div class="grid g2" style="margin-top:16px">
   <div class="card"><h2>Metrics by model</h2><div class="chart tall"><canvas id="m1"></canvas></div></div>
   <div class="card"><h2>Confusion matrix</h2><div class="tabs" id="cm-tabs">${names.map(n => `<button data-n="${esc(n)}" class="${n === sel ? "on" : ""}">${esc(n)}</button>`).join("")}</div><div id="cm"></div></div>
   <div class="card"><h2>Precision-recall curves</h2><div class="chart tall"><canvas id="m2"></canvas></div></div>
   <div class="card"><h2>ROC curves</h2><div class="chart tall"><canvas id="m3"></canvas></div></div>
  </div>
  <div class="card" style="margin-top:16px"><h2>Performance by source (selected model, time-based test set)</h2><p class="sub">Small samples: treat as indicative. PaySim is easy because type and amount separate fraud well; the two card/e-commerce sources only have amount and hour to work with.</p>
   <div class="tablewrap"><table><thead><tr><th>Source</th><th class="num">Test rows</th><th class="num">Fraud</th><th class="num">ROC-AUC</th><th class="num">PR-AUC</th><th class="num">Recall</th><th class="num">False-positive rate</th></tr></thead><tbody>
   ${Object.entries(m.per_source).map(([s, r]) => `<tr><td>${s}</td><td class="num">${r.n_test}</td><td class="num">${r.fraud_test}</td><td class="num">${r.roc_auc.toFixed(3)}</td><td class="num">${r.pr_auc.toFixed(3)}</td><td class="num">${pct(r.recall)}</td><td class="num">${pct(r.false_positive_rate)}</td></tr>`).join("")}</tbody></table></div></div>
  <div class="card" style="margin-top:16px"><h2>What this would mean at real-world fraud rates (illustrative)</h2><p class="sub">The sample is 50% fraud, so precision above is far too rosy. Using each source's recall and false-positive rate with its approximate real fraud prevalence:</p>
   <div class="tablewrap"><table><thead><tr><th>Source</th><th class="num">Real fraud rate</th><th class="num">Precision at real rate</th><th class="num">Alerts per 1,000 txns</th><th class="num">False alarms per true fraud caught</th></tr></thead><tbody>
   ${m.base_rate_projection.map(r => `<tr><td>${r.source}</td><td class="num">${pct(r.real_base_rate, 2)}</td><td class="num">${r.precision_at_real_rate == null ? "n/a" : pct(r.precision_at_real_rate, 1)}</td><td class="num">${r.alerts_per_1000.toFixed(0)}</td><td class="num">${r.false_alarms_per_true_fraud == null ? "n/a" : r.false_alarms_per_true_fraud.toFixed(0)}</td></tr>`).join("")}</tbody></table></div>
   <div class="hl">Real prevalence figures are approximate values from the public documentation of the full datasets. This is a projection from small test samples, not a measured result.</div></div>`;
  document.querySelectorAll("#proto button").forEach(b => (b.onclick = () => { PROTO = b.dataset.p; route(); }));
  mkChart("m1", { type: "bar", data: { labels: names, datasets: [["precision", "Precision", C.blue], ["recall", "Recall", C.safe], ["f1", "F1", C.amber], ["pr_auc", "PR-AUC", C.purple]].map(([k, l, c]) => ({ label: l, data: names.map(n => +P.models[n][k].toFixed(4)), backgroundColor: c })) }, options: { scales: { y: { min: 0, max: 1 } } } });
  const line = (key, xl, yl) => ({ type: "scatter", data: { datasets: names.map((n, i) => ({ label: n, data: P.models[n][key].map(([x, y]) => ({ x, y })), showLine: true, borderColor: colors[i], pointRadius: 0, borderWidth: n === sel ? 3 : 1.5 })) }, options: { scales: { x: { min: 0, max: 1, title: { display: true, text: xl } }, y: { min: 0, max: 1, title: { display: true, text: yl } } } } });
  mkChart("m2", line("pr_curve", "Recall", "Precision")); mkChart("m3", line("roc_curve", "False positive rate", "True positive rate"));
  const showCM = n => {
    const r = P.models[n], c = r.confusion_matrix, pos = c.tp + c.fn;
    document.querySelectorAll("#cm-tabs button").forEach(b => b.classList.toggle("on", b.dataset.n === n));
    $("#cm").innerHTML = `<div class="cm"><div class="h"></div><div class="h">Predicted legit</div><div class="h">Predicted fraud</div>
      <div class="h">Actually legit</div><div class="tn"><strong>${nf.format(c.tn)}</strong><small>True negatives</small></div><div class="fp"><strong>${nf.format(c.fp)}</strong><small>False alarms</small></div>
      <div class="h">Actually fraud</div><div class="fn"><strong>${nf.format(c.fn)}</strong><small>Missed fraud</small></div><div class="tp"><strong>${nf.format(c.tp)}</strong><small>Fraud caught</small></div></div>
      <div class="hl">At threshold ${r.threshold.toFixed(3)}, ${esc(n)} caught <strong>${c.tp} of ${pos}</strong> fraud cases (${pct(r.recall)}) and raised <strong>${c.fp}</strong> false alarms among ${nf.format(c.tn + c.fp)} legitimate transactions.</div>`;
  };
  document.querySelectorAll("#cm-tabs button").forEach(b => (b.onclick = () => showCM(b.dataset.n)));
  showCM(sel);
}

/* ------------------------------------------------------------------ Reports */
async function pageReports() {
  const s = await api("/api/summary");
  view.innerHTML = `<p class="sub">Every report is generated live from the loaded dataset and saved model results.</p>
  <div class="grid g3">
   <div class="card"><h2>Filtered transactions</h2><p class="sub">Choose what to include.</p>
    <label class="f">Source<select id="r-source">${["ALL", "PaySim", "IEEE-CIS", "ULB"].map(t => `<option>${t}</option>`).join("")}</select></label><br>
    <label class="f">Actual label<select id="r-label"><option value="all">All</option><option value="fraud">Fraud only</option><option value="legit">Legitimate only</option></select></label><br>
    <label class="f">Model verdict<select id="r-flag"><option value="all">All</option><option value="flagged">Flagged by model</option></select></label><br>
    <button class="btn" id="r1">Download CSV</button></div>
   <div class="card"><h2>Confirmed fraud cases</h2><p class="sub">All ${nf.format(s.fraud_cases)} labelled fraud transactions with model risk scores.</p><button class="btn" id="r2">Download CSV</button></div>
   <div class="card"><h2>Model comparison</h2><p class="sub">All four models under both evaluation protocols: precision, recall, F1, ROC-AUC, PR-AUC, CV PR-AUC and confusion-matrix counts.</p><button class="btn" id="r3">Download CSV</button></div>
  </div>
  <div class="card" style="margin-top:16px"><h2>Report snapshot</h2><dl class="kv">
   <dt>Dataset</dt><dd>Mendeley harmonized package: ${nf.format(s.total_transactions)} transactions from PaySim, IEEE-CIS and ULB (balanced sample)</dd>
   <dt>Fraud in sample</dt><dd>${nf.format(s.fraud_cases)} cases (${pct(s.fraud_share, 0)}, a sampling choice)</dd>
   <dt>Final model</dt><dd>${esc(s.model.name)}: precision ${pct(s.model.precision)}, recall ${pct(s.model.recall)}, F1 ${s.model.f1.toFixed(3)}, PR-AUC ${s.model.pr_auc.toFixed(3)} (time-based test set)</dd></dl></div>`;
  $("#r1").onclick = () => download("/api/export/transactions?" + new URLSearchParams({ source: $("#r-source").value, label: $("#r-label").value, model_flag: $("#r-flag").value }), "fraudlens_transactions.csv");
  $("#r2").onclick = () => download("/api/export/fraud-cases", "fraudlens_confirmed_fraud_cases.csv");
  $("#r3").onclick = () => download("/api/export/model-metrics", "fraudlens_model_comparison.csv");
}

/* ------------------------------------------------------------------ Dataset */
async function pageDataset() {
  const d = await api("/api/dataset-info"), ds = d.dataset, ig = d.integrity;
  const kv = o => Object.entries(o).map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join("");
  view.innerHTML = `
  <div class="banner">${esc(d.caveats[0])} ${esc(d.caveats[1])}</div>
  <div class="grid g2">
   <div class="card"><h2>Dataset</h2><dl class="kv">${kv({ Name: ds.name, Rows: nf.format(ds.rows), "Fraud rows": nf.format(ds.fraud_rows), ...Object.fromEntries(Object.entries(ds.per_source).map(([s, v]) => [s, `${nf.format(v.rows)} rows, ${nf.format(v.fraud)} fraud`])), "Train (earlier 80%)": `${nf.format(d.split.train.rows)} rows, ${d.split.train.fraud} fraud`, "Test (later 20%)": `${nf.format(d.split.test.rows)} rows, ${d.split.test.fraud} fraud` })}</dl></div>
   <div class="card"><h2>Split integrity checks (computed at training)</h2><dl class="kv">${kv({ "Time split covers all 5,000 rows": ig.time_split_covers_all_rows ? "Yes" : "No", "Random split covers all 5,000 rows": ig.random_split_covers_all_rows ? "Yes" : "No", "Time split chronological per source": Object.entries(ig.time_split_is_chronological_per_source).map(([k, v]) => `${k}: ${v ? "yes" : "no"}`).join(", "), "Time split train/test overlap": ig.time_split_train_test_key_overlap, "Random split train/test overlap": ig.random_split_train_test_key_overlap, "Duplicate rows in dataset": ig.duplicate_rows_in_harmonized })}</dl></div>
   <div class="card"><h2>Model inputs</h2><p>${d.model_features.map(f => `<span class="badge b-sel">${esc(f)}</span>`).join(" ")}</p><p class="sub">Raw inputs: ${d.raw_features.map(esc).join(", ")}</p>
     <h2 style="margin-top:18px">Transformed / excluded</h2><dl class="kv">${kv(d.excluded_or_transformed)}</dl>
     <h2 style="margin-top:18px">Types by source</h2><dl class="kv">${kv(Object.fromEntries(Object.entries(d.types_by_source).map(([k, v]) => [k, v.join(", ")])))}</dl></div>
   <div class="card"><h2>Caveats</h2><ul class="clean">${d.caveats.map(c => `<li>${esc(c)}</li>`).join("")}</ul>
     <h2 style="margin-top:18px">Environment</h2><dl class="kv">${kv({ ...d.versions, "Trained at": d.trained_at })}</dl></div>
  </div>`;
}

/* ------------------------------------------------------------------ AI Explainability */
const EX = { txId: null, ctx: null, chat: [], busy: false, token: 0, total: 0, llm: null };
const EX_GROUPS = { all: { label: "All transactions" }, tp: { label: "Flagged and actually fraud (true positives)", lab: "fraud", flag: "flagged" },
  fp: { label: "Flagged but actually legitimate (false alarms)", lab: "legit", flag: "flagged" }, fn: { label: "Fraud the model missed", lab: "fraud", flag: "not_flagged" },
  tn: { label: "Legitimate and not flagged", lab: "legit", flag: "not_flagged" } };
const SUGGESTED = ["Why was this transaction flagged as suspicious?", "Which features contributed most to its fraud score?", "What is the difference between fraud probability and the decision threshold?",
  "Why might a legitimate transaction receive a fraud alert?", "What does this model's precision or recall mean?", "How confident should I be in this prediction?", "What additional information would help investigate this transaction?"];
const EX_SEL = { group: "all", source: "ALL", split: "test" };

async function pageExplain() {
  const [sum, st] = await Promise.all([api("/api/summary"), api("/api/chat/status")]);
  EX.total = sum.total_transactions; EX.llm = st;
  view.innerHTML = `
  <div class="card"><h2>Why was this transaction flagged?</h2>
   <p class="sub">Pick a transaction from the dataset. The prediction comes from the saved model, and each explanation is calculated for that transaction only.</p>
   <div class="picker">
    <label class="f">Show<select id="x-group">${Object.entries(EX_GROUPS).map(([k, g]) => `<option value="${k}">${g.label}</option>`).join("")}</select></label>
    <label class="f">Source<select id="x-source"><option value="ALL">All sources</option><option>PaySim</option><option>IEEE-CIS</option><option>ULB</option></select></label>
    <label class="f">Rows from<select id="x-split"><option value="test">Test set (unseen by the model)</option><option value="train">Training set</option><option value="all">All rows</option></select></label>
    <label class="f">Go to transaction ID (1-${nf.format(EX.total)})<span style="display:flex;gap:6px"><input id="x-id" type="number" min="1" max="${EX.total}" step="1" style="width:100%"><button class="btn" id="x-go" type="button">Load</button></span></label>
    <label class="f wide">Transaction<select id="x-pick" aria-describedby="x-count"></select></label>
    <div class="sub wide" id="x-count" style="margin:0"></div>
   </div></div>
  <div id="ex-body" style="margin-top:16px"></div>`;
  $("#x-group").value = EX_SEL.group; $("#x-source").value = EX_SEL.source; $("#x-split").value = EX_SEL.split;
  ["group", "source", "split"].forEach(k => $("#x-" + k).addEventListener("change", e => { EX_SEL[k] = e.target.value; loadExList(true); }));
  $("#x-pick").addEventListener("change", e => selectTx(+e.target.value));
  $("#x-go").onclick = () => {
    const v = $("#x-id").value.trim(), n = Number(v);
    if (!v || !Number.isInteger(n) || n < 1 || n > EX.total) { $("#ex-body").innerHTML = `<div class="banner err">Enter a whole number between 1 and ${nf.format(EX.total)}.</div>`; return; }
    selectTx(n);
  };
  $("#x-id").addEventListener("keydown", e => { if (e.key === "Enter") $("#x-go").click(); });
  await loadExList(true);
}
async function loadExList(selectFirst) {
  const g = EX_GROUPS[EX_SEL.group], p = new URLSearchParams({ page_size: "100", sort_by: "risk_score", sort_dir: g.flag === "not_flagged" ? "asc" : "desc", split: EX_SEL.split, source: EX_SEL.source });
  if (g.lab) p.set("label", g.lab); if (g.flag) p.set("model_flag", g.flag);
  const pick = $("#x-pick"); pick.disabled = true; pick.innerHTML = `<option>Loading...</option>`;
  try {
    const d = await api("/api/transactions?" + p);
    if (!d.rows.length) { pick.innerHTML = `<option value="">No matching transactions</option>`; $("#x-count").textContent = "No transactions match these choices."; $("#ex-body").innerHTML = `<div class="card empty">No transactions match. Try another group, source or set.</div>`; return; }
    pick.innerHTML = d.rows.map(r => `<option value="${r.tx_id}">#${r.tx_id} · ${esc(r.Source)} · ${esc(r.TransactionType)} · amount ${nf.format(r.Amount)} · table score ${(r.risk_score * 100).toFixed(1)}%</option>`).join("");
    $("#x-count").textContent = `Showing ${d.rows.length} of ${nf.format(d.total)} matching transactions, ordered by model score.`;
    pick.disabled = false;
    if (selectFirst) { const keep = EX.txId && d.rows.some(r => r.tx_id === EX.txId) ? EX.txId : d.rows[0].tx_id; pick.value = keep; selectTx(keep); }
  } catch (e) { pick.innerHTML = `<option value="">Could not load</option>`; $("#ex-body").innerHTML = `<div class="banner err">${esc(e.message)}</div>`; }
}
async function selectTx(id) {
  const my = ++EX.token; $("#ex-body").innerHTML = `<div class="loading"><div class="spin"></div>Scoring transaction ${id} and calculating contributions...</div>`;
  try {
    const ctx = await api("/api/explain/" + id);
    if (my !== EX.token) return;
    EX.txId = id; EX.ctx = ctx; EX.busy = false;
    EX.chat = [{ role: "sys", text: `New conversation for transaction #${id} (${ctx.transaction.source}, ${ctx.transaction.transaction_type}). Answers use this transaction's model output only.` }];
    renderExplain();
  } catch (e) { if (my === EX.token) $("#ex-body").innerHTML = `<div class="banner err"><strong>Could not load transaction ${id}.</strong> ${esc(e.message)}</div>`; }
}
function renderExplain() {
  const c = EX.ctx, t = c.transaction, p = c.prediction, e = c.explanation, sm = c.source_metrics;
  charts.forEach(ch => ch.destroy()); charts = [];
  const lvl = p.risk_level.toLowerCase();
  const outcomeText = { true_positive: "Model and record agree: fraud (true positive)", false_positive: "Model flagged it, record says legitimate (false alarm)", false_negative: "Record says fraud, model did not flag it (missed fraud)", true_negative: "Model and record agree: legitimate" }[c.outcome];
  const typeLabel = t.source === "IEEE-CIS" ? "Product code (ProductCD)" : "Transaction type";
  $("#ex-body").innerHTML = `
  <div class="two-col">
   <div>
    <div class="card"><h2>Selected transaction</h2><dl class="kv kv-sm" style="margin-top:10px">
      <dt>Transaction ID</dt><dd>#${t.tx_id}</dd><dt>Data source</dt><dd>${esc(t.source)}</dd><dt>${typeLabel}</dt><dd>${esc(t.transaction_type)}</dd>
      <dt>Amount</dt><dd>${nf.format(t.amount)} <span class="sub">(${esc(t.source)}'s own units)</span></dd><dt>Hour of day</dt><dd>${String(t.hour_of_day).padStart(2, "0")}:00</dd>
      <dt>Dataset set</dt><dd>${t.in_training_data ? "Training set (the model has seen this row)" : "Test set (held out from training)"}</dd>
      <dt>Actual label</dt><dd><span class="badge ${t.actual_label ? "b-fraud" : "b-ok"}">${t.actual_label ? "Fraud" : "Legitimate"}</span> <span class="sub">historical record</span></dd></dl></div>
    <div class="card" style="margin-top:16px"><h2>Fraud risk summary</h2>
     <div class="verdict ${lvl}" style="margin-top:10px"><div>Model prediction: <strong>${p.predicted_class}</strong> · ${p.risk_level} risk</div>
      <div class="big">${(p.probability * 100).toFixed(1)}%</div>
      <div class="meter" role="img" aria-label="Fraud score ${(p.probability * 100).toFixed(1)} percent; decision threshold ${(p.threshold * 100).toFixed(1)} percent"><i style="width:${Math.min(100, p.probability * 100)}%"></i><b style="left:${Math.min(100, p.threshold * 100)}%" title="Decision threshold"></b></div>
      <small>Fraud score from ${esc(p.model)}. Black marker = decision threshold ${(p.threshold * 100).toFixed(1)}%. Score is ${Math.abs(p.margin * 100).toFixed(1)} points ${p.margin >= 0 ? "above" : "below"} it.</small></div>
     <div class="cmp"><div>Model prediction<strong>${p.predicted_class}</strong></div><div>Actual label (history)<strong>${t.actual_label ? "Fraud" : "Legitimate"}</strong></div></div>
     <p class="sub" style="margin:10px 0 0">${esc(outcomeText)}.${t.in_training_data ? " This row was part of training, so the match says less about unseen data." : ""}</p></div>
   </div>
   <div class="card"><h2>Explanation</h2>
    ${e ? `<div class="hl" style="margin:10px 0 14px">${esc(c.summary)}</div>
     <div class="chart" style="height:${70 + e.contributions.length * 52}px"><canvas id="ex-chart" role="img" aria-label="Bar chart of each input's contribution to this transaction's fraud score"></canvas></div>
     <div class="tablewrap" style="margin-top:12px"><table><thead><tr><th>Input</th><th>Value</th><th>Effect on fraud score</th><th class="num">Points</th></tr></thead><tbody>
      ${e.contributions.map(x => `<tr><td>${esc(x.label)}</td><td>${esc(x.display_value)}</td><td>${x.direction === "increases" ? "Pushes score up" : x.direction === "decreases" ? "Pushes score down" : "No effect"}</td><td class="num">${x.contribution >= 0 ? "+" : "-"}${Math.abs(x.contribution * 100).toFixed(1)}</td></tr>`).join("")}</tbody></table></div>
     <p class="sub" style="margin-top:10px">Baseline: ${esc(e.baseline_description)} scores ${(e.baseline_score * 100).toFixed(1)}%. Baseline + contributions = ${(p.probability * 100).toFixed(1)}%. ${esc(e.method)}. Source (${esc(t.source)}) is held fixed. These contributions describe the model's reasoning, not necessarily the real-world cause of fraud.</p>`
    : `<div class="banner err" style="margin-top:10px"><strong>Feature contributions are unavailable for this transaction.</strong> ${esc(c.explanation_error || "")}<br>The prediction above is still the real model output.</div>`}
   </div>
  </div>
  <div class="chat-grid">
   <div class="card"><h2>Ask FraudLens AI</h2>
    <p class="sub"><span class="badge ${EX.llm.llm_configured ? "b-sel" : "b-ok"}">${EX.llm.llm_configured ? "LLM-assisted answers" : "Built-in answers"}</span> ${EX.llm.llm_configured ? "Answers are drafted from this transaction's verified model output; some questions are always answered by the built-in rules." : "Answers are generated on the server from this transaction's model output. No external AI service is used."}</p>
    <div class="chat-log" id="chat-log" role="log" aria-live="polite" aria-label="Conversation with FraudLens AI" tabindex="0"></div>
    <form class="chat-form" id="chat-form"><label class="sr" for="chat-input" style="position:absolute;left:-999px">Your question</label>
     <input id="chat-input" type="text" maxlength="500" autocomplete="off" placeholder="Ask why this transaction was flagged..."><button class="btn" id="chat-send" type="submit">Send</button>
     <button class="btn ghost" id="chat-clear" type="button">New conversation</button></form>
    <div class="chips" id="chips">${SUGGESTED.map((q, i) => `<button type="button" data-q="${i}">${esc(q)}</button>`).join("")}</div>
   </div>
   <div class="card"><h2>Model limitations</h2><ul class="clean">
    <li><strong>A flag is not proof.</strong> A prediction means "worth investigating". The model sees only source, type, amount and hour of day.</li>
    <li><strong>Performance differs by source.</strong> Held-out ROC-AUC: ${Object.entries(c.all_source_auc).map(([s, v]) => `${esc(s)} ${v.roc_auc.toFixed(2)}`).join(", ")}. This transaction is from ${esc(sm.source)} (${sm.n_test} test rows).</li>
    <li><strong>Balanced benchmark sample.</strong> About ${(c.dataset.fraud_share * 100).toFixed(0)}% of rows are fraud by design; real fraud is far rarer, so scores are not real-world probabilities.</li>
    <li><strong>Explanations are not causes.</strong> They show how the model reached its score.</li>
    <li>PaySim is simulated data, not real UPI transactions.</li></ul></div>
  </div>`;
  if (e) {
    const cs = e.contributions, col = cs.map(x => x.contribution >= 0 ? C.fraud : C.safe);
    mkChart("ex-chart", { type: "bar", data: { labels: cs.map(x => `${x.label} = ${x.display_value}`), datasets: [{ data: cs.map(x => +(x.contribution * 100).toFixed(2)), backgroundColor: col }] },
      options: { indexAxis: "y", plugins: { legend: { display: false }, tooltip: { callbacks: { label: i => `${i.parsed.x >= 0 ? "+" : ""}${i.parsed.x} score points` } } }, scales: { x: { title: { display: true, text: "Score points (right = pushes fraud score up, left = pushes it down)" } } } } });
  }
  renderChat();
  $("#chat-form").onsubmit = ev => { ev.preventDefault(); sendChat($("#chat-input").value); };
  $("#chat-clear").onclick = () => { EX.chat = [{ role: "sys", text: `New conversation for transaction #${EX.txId}.` }]; EX.busy = false; renderChat(); $("#chat-input").focus(); };
  $("#chips").querySelectorAll("button").forEach(b => (b.onclick = () => sendChat(SUGGESTED[+b.dataset.q])));
}
function renderChat() {
  const log = $("#chat-log"); if (!log) return;
  log.innerHTML = EX.chat.map(m => m.role === "pending" ? `<div class="msg"><span class="dots">FraudLens AI is thinking</span></div>`
    : `<div class="msg ${m.role === "user" ? "user" : m.role === "sys" ? "sys" : m.role === "error" ? "err" : ""}">${esc(m.text)}${m.meta ? `<small>${esc(m.meta)}</small>` : ""}</div>`).join("");
  log.scrollTop = log.scrollHeight;
  const dis = EX.busy; $("#chat-send").disabled = dis; $("#chips").querySelectorAll("button").forEach(b => (b.disabled = dis));
}
async function sendChat(text) {
  text = (text || "").trim();
  if (!text || EX.busy) { if (!text) toast("Type a question first.", true); return; }
  if (text.length > 500) { toast("Please keep the question under 500 characters.", true); return; }
  const id = EX.txId, my = EX.token;
  const history = EX.chat.filter(m => m.role === "user" || m.role === "assistant").slice(-6).map(m => ({ role: m.role, content: m.text }));
  EX.chat.push({ role: "user", text }); EX.chat.push({ role: "pending" }); EX.busy = true; $("#chat-input").value = ""; renderChat();
  try {
    const r = await api("/api/chat", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ tx_id: id, message: text, history }) });
    if (my !== EX.token || id !== EX.txId) return;                      // user switched transaction while waiting: drop the stale answer
    EX.chat = EX.chat.filter(m => m.role !== "pending");
    EX.chat.push({ role: "assistant", text: r.answer, meta: `${r.mode === "llm-assisted" ? "LLM-assisted" : "Built-in"} answer about transaction #${r.tx_id}` + (r.llm_error ? ` · ${r.llm_error}` : "") });
  } catch (e) {
    if (my !== EX.token || id !== EX.txId) return;
    EX.chat = EX.chat.filter(m => m.role !== "pending"); EX.chat.push({ role: "error", text: "Sorry, I could not answer that. " + e.message });
  }
  EX.busy = false; renderChat(); const inp = $("#chat-input"); if (inp) inp.focus();
}

/* ------------------------------------------------------------------ Router */
const ROUTES = { overview: ["Overview", pageOverview], transactions: ["Transaction explorer", pageTransactions], detect: ["Fraud detection", pageDetect],
  analytics: ["Analytics", pageAnalytics], performance: ["Model performance", pagePerformance], explain: ["AI Explainability", pageExplain], reports: ["Reports", pageReports], dataset: ["Dataset and model", pageDataset] };
async function route() {
  const key = (location.hash.replace(/^#\/?/, "") || "overview");
  const [title, fn] = ROUTES[key] || ROUTES.overview;
  document.querySelectorAll("#nav a").forEach(a => a.classList.toggle("active", a.dataset.route === (ROUTES[key] ? key : "overview")));
  $("#page-title").textContent = title; document.title = title + " - FraudLens AI";
  $("#sidebar").classList.remove("open");
  charts.forEach(c => c.destroy()); charts = [];
  loading();
  try { await fn(); } catch (e) { errorBox(e); }
}

/* ------------------------------------------------------------------ Startup */
$("#menu").onclick = () => $("#sidebar").classList.toggle("open");
window.addEventListener("hashchange", route);
async function boot() {
  chartDefaults();
  try {
    const h = await api("/api/health");
    $("#side-foot").innerHTML = h.status === "ok"
      ? `<span class="dot"></span>Backend online<br>Model: ${esc(h.model)}<br>${nf.format(h.rows_loaded)} transactions`
      : `<span class="dot bad"></span>Backend error`;
    if (h.status !== "ok") { view.innerHTML = `<div class="banner err"><strong>The backend started but is not ready.</strong> ${esc(h.error)}</div>`; return; }
    const s = await api("/api/summary");
    $("#data-chip").textContent = "Balanced benchmark sample (50% fraud)";
    $("#data-chip").title = "Real-world fraud rates are far lower; see the Dataset page.";
    SCHEMA = await api("/api/dataset-info");
  } catch (e) {
    $("#side-foot").innerHTML = `<span class="dot bad"></span>Backend offline`;
    view.innerHTML = `<div class="banner err"><strong>Backend not reachable.</strong> ${esc(e.message)}</div>`; return;
  }
  route();
}
boot();
