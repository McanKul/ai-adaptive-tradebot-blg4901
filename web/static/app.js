"use strict";

const $ = (sel) => document.querySelector(sel);

// Attributes that are read-only as DOM properties and must be set via
// setAttribute (assigning them directly throws under "use strict").
const ATTR_ONLY = new Set(["list"]);

const el = (tag, props = {}, ...kids) => {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (v == null) continue;
    if (k === "dataset") Object.assign(n.dataset, v);
    else if (ATTR_ONLY.has(k)) n.setAttribute(k, v);
    else n[k] = v;
  }
  for (const c of kids) if (c != null) n.append(c);
  return n;
};

let SCHEMA = {};        // { commandName: {name, params:[...]} }
let CURRENT_CMD = null;
let evtSource = null;
let DATASETS = [];      // [{symbol, days, start, end}]
let streamTarget = "#console";

const COMMAND_HELP = {
  "backtest": "Run a single backtest (optionally with cross-validation).",
  "walk-forward": "Walk-forward cross-validation (alias for backtest --cv-method walk_forward).",
  "live": "Start LIVE trading — places real orders. Requires a promotion-gate stamp.",
  "dry-run": "Paper trading on live market data — no real orders.",
  "sweep": "Grid parameter sweep with optional CV / Hyperband.",
  "validate": "Validate a config file (optionally with strict real-money checks).",
};

// ──────────────────────────────────────────────────────────────────────
// Bootstrap
// ──────────────────────────────────────────────────────────────────────
async function init() {
  setupTabs();
  setupActions();

  const [schemaRes, stratRes, filesRes] = await Promise.all([
    fetch("/api/schema").then((r) => r.json()),
    fetch("/api/strategies").then((r) => r.json()),
    fetch("/api/files").then((r) => r.json()),
  ]);

  schemaRes.commands.forEach((c) => (SCHEMA[c.name] = c));
  populateDatalist("strategyOptions", stratRes.strategies || []);
  buildFileDatalists(filesRes.files || []);
  populateFileSelect(filesRes.files || []);

  renderCommandBar(schemaRes.commands.map((c) => c.name));
  selectCommand(schemaRes.commands[0].name);

  await loadDatasets();
  loadEnv();

  // deep-link to a tab via #hash (e.g. /#data)
  const h = location.hash.replace("#", "");
  if (h && document.getElementById("tab-" + h)) switchTab(h);
}

async function loadDatasets() {
  try {
    const res = await fetch("/api/datasets").then((r) => r.json());
    DATASETS = res.datasets || [];
  } catch (e) {
    DATASETS = [];
  }
  populateDatalist("datasetOptions", DATASETS.map((d) => d.symbol));
  renderDatasetChips();
  renderDatasetTable();
}

function renderDatasetChips() {
  const wrap = $("#datasetToggle");
  const chips = $("#datasetChips");
  chips.innerHTML = "";
  if (!DATASETS.length) { wrap.hidden = true; return; }
  wrap.hidden = false;
  DATASETS.forEach((d) => {
    chips.append(el("button", {
      className: "chip", type: "button",
      textContent: d.symbol,
      title: `${d.days} days (${d.start} → ${d.end})`,
      onclick: () => useSymbol(d.symbol),
    }));
  });
}

function renderDatasetTable() {
  const tbody = document.querySelector("#datasetTable tbody");
  tbody.innerHTML = "";
  if (!DATASETS.length) {
    tbody.append(el("tr", {}, el("td", { colSpan: 4, className: "muted-cell", textContent: "No datasets yet — fetch some below." })));
    return;
  }
  DATASETS.forEach((d) => {
    const useBtn = el("button", { className: "btn ghost small", type: "button", textContent: "Use in Run", onclick: () => useSymbol(d.symbol) });
    tbody.append(el("tr", {},
      el("td", { textContent: d.symbol }),
      el("td", { textContent: String(d.days) }),
      el("td", { textContent: `${d.start} → ${d.end}` }),
      el("td", {}, useBtn)));
  });
}

// set the symbol field of the current command and jump to the Run tab
function useSymbol(sym) {
  const field = document.querySelector('#settingsForm [data-dest="symbol"]');
  if (field) { field.value = sym; updatePreview(); }
  document.querySelectorAll(".chip").forEach((c) =>
    c.classList.toggle("active", c.textContent === sym));
  switchTab("run");
}

function populateDatalist(id, items) {
  const dl = document.getElementById(id);
  if (!dl) return;
  dl.innerHTML = "";
  items.forEach((v) => dl.append(el("option", { value: v })));
}

// Per-purpose file pickers so each field only suggests relevant files.
const FILE_FILTERS = {
  "files-realism": (f) => /realism/i.test(f),
  "files-grid": (f) => /grid/i.test(f),
  "files-composite": (f) => /composite/i.test(f),
  "files-config": (f) => /\.(ya?ml|json)$/i.test(f) && !/grid|realism|composite|\.github/i.test(f),
};

function fileListId(dest) {
  return {
    realism_config: "files-realism",
    param_grid: "files-grid",
    composite_spec: "files-composite",
    config: "files-config",
  }[dest] || "fileOptions";
}

function buildFileDatalists(files) {
  populateDatalist("fileOptions", files);
  for (const [id, filt] of Object.entries(FILE_FILTERS)) {
    if (!document.getElementById(id)) document.body.append(el("datalist", { id }));
    populateDatalist(id, files.filter(filt));
  }
}

function populateFileSelect(files) {
  const sel = $("#fileSelect");
  sel.innerHTML = "";
  files.forEach((f) => sel.append(el("option", { value: f, textContent: f })));
}

// ──────────────────────────────────────────────────────────────────────
// Command bar + form
// ──────────────────────────────────────────────────────────────────────
function renderCommandBar(names) {
  const bar = $("#commandBar");
  bar.innerHTML = "";
  names.forEach((name) => {
    bar.append(
      el("button", {
        className: "cmd-btn",
        type: "button",
        textContent: name,
        onclick: () => selectCommand(name),
        dataset: { cmd: name },
      })
    );
  });
}

function selectCommand(name) {
  CURRENT_CMD = name;
  document.querySelectorAll(".cmd-btn").forEach((b) =>
    b.classList.toggle("active", b.dataset.cmd === name)
  );
  $("#commandHelp").textContent = COMMAND_HELP[name] || "";
  renderForm(SCHEMA[name]);
  updatePreview();
}

function renderForm(command) {
  const form = $("#settingsForm");
  form.innerHTML = "";

  // group params by their "group" label, preserving first-seen order
  const groups = new Map();
  command.params.forEach((p) => {
    if (!groups.has(p.group)) groups.set(p.group, []);
    groups.get(p.group).push(p);
  });

  for (const [title, params] of groups) {
    const fields = el("div", { className: "fields" });
    params.forEach((p) => fields.append(renderField(p)));
    form.append(el("div", { className: "group" }, el("h3", { className: "group-title", textContent: title }), fields));
  }

  form.addEventListener("input", updatePreview);
  form.addEventListener("change", updatePreview);
}

function renderField(p) {
  const wrap = el("div", { className: "field" });
  const label = el("label", { textContent: prettyName(p) });
  if (p.required) label.append(el("span", { className: "req", textContent: "*" }));
  wrap.append(label);

  let control;
  if (p.type === "bool" && p.flags && p.flags.true && p.flags.false) {
    // tri-state (e.g. --tick-exit / --no-tick-exit)
    control = el("select", { className: "select", dataset: { dest: p.dest, kind: "tristate" } });
    const def = p.default ? "on" : "off";
    control.append(el("option", { value: "", textContent: `Default (${def})` }));
    control.append(el("option", { value: "on", textContent: `On  (${p.flags.true})` }));
    control.append(el("option", { value: "off", textContent: `Off (${p.flags.false})` }));
  } else if (p.type === "bool") {
    // single store_true flag
    const cb = el("input", { type: "checkbox", dataset: { dest: p.dest, kind: "bool" } });
    cb.checked = p.default === true;
    control = el("label", { className: "checkbox-row" }, cb, el("span", { textContent: p.flags && p.flags.true ? p.flags.true : "enable" }));
  } else if (p.type === "choice") {
    control = el("select", { className: "select", dataset: { dest: p.dest, kind: "value" } });
    control.append(el("option", { value: "", textContent: `— default${p.default != null ? " (" + p.default + ")" : ""} —` }));
    (p.choices || []).forEach((c) => control.append(el("option", { value: c, textContent: c })));
  } else if (p.widget === "strategy" || p.widget === "file") {
    control = el("input", {
      className: "input", type: "text",
      list: p.widget === "strategy" ? "strategyOptions" : fileListId(p.dest),
      placeholder: p.default != null ? String(p.default) : "",
      dataset: { dest: p.dest, kind: "value" },
    });
  } else {
    const attrs = {
      className: "input", type: "text",
      placeholder: p.default != null ? String(p.default) : "",
      dataset: { dest: p.dest, kind: "value" },
    };
    if (p.dest === "symbol") attrs.list = "datasetOptions";
    control = el("input", attrs);
  }
  wrap.append(control);

  if (p.help) wrap.append(el("div", { className: "desc", textContent: p.help }));
  if (p.default != null && p.type !== "bool")
    wrap.append(el("div", { className: "deflt", textContent: `default: ${p.default}` }));
  return wrap;
}

function prettyName(p) {
  const opt = p.option || (p.flags && (p.flags.true || p.flags.false)) || ("--" + p.dest);
  return opt;
}

// collect form values keyed by dest
function collectValues() {
  const values = {};
  document.querySelectorAll("#settingsForm [data-dest]").forEach((node) => {
    const dest = node.dataset.dest;
    if (node.dataset.kind === "bool") values[dest] = node.checked;
    else values[dest] = node.value;
  });
  return values;
}

// ──────────────────────────────────────────────────────────────────────
// Preview (debounced)
// ──────────────────────────────────────────────────────────────────────
let previewTimer = null;
function updatePreview() {
  clearTimeout(previewTimer);
  previewTimer = setTimeout(async () => {
    const body = { command: CURRENT_CMD, values: collectValues() };
    try {
      const res = await fetch("/api/preview", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      }).then((r) => r.json());
      $("#preview").textContent = res.command || "python app.py …";
      const errBox = $("#formErrors");
      if (res.errors && res.errors.length) {
        errBox.textContent = "⚠ " + res.errors.join(", ");
        errBox.classList.remove("ok");
      } else {
        errBox.textContent = "";
      }
    } catch (e) {
      $("#preview").textContent = "(preview unavailable)";
    }
  }, 200);
}

// ──────────────────────────────────────────────────────────────────────
// Run / Stop / streaming
// ──────────────────────────────────────────────────────────────────────
function setupActions() {
  $("#runBtn").onclick = runCommand;
  $("#stopBtn").onclick = stopCommand;
  $("#clearBtn").onclick = () => ($("#console").innerHTML = "");

  $("#fileLoad").onclick = loadFile;
  $("#fileSave").onclick = saveFile;
  $("#envLoad").onclick = loadEnv;
  $("#envSave").onclick = saveEnv;

  $("#datasetsRefresh").onclick = loadDatasets;
  $("#fetchBtn").onclick = fetchData;
  $("#fetchStopBtn").onclick = stopCommand;
}

async function fetchData() {
  const body = {
    symbol: $("#fetchSymbol").value,
    start: $("#fetchStart").value,
    end: $("#fetchEnd").value,
    market_type: $("#fetchMarket").value,
    data_type: $("#fetchDataType").value,
  };
  const res = await fetch("/api/fetch", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  }).then((r) => r.json());

  if (!res.ok) {
    msg("#fetchMsg", "⚠ " + (res.errors || ["failed to start"]).join(", "), false);
    return;
  }
  msg("#fetchMsg", "", true);
  $("#dataConsole").innerHTML = "";
  $("#fetchBtn").disabled = true;
  $("#fetchStopBtn").disabled = false;
  setPill(true);
  openStream({
    target: "#dataConsole",
    onDone: (rc) => {
      $("#fetchBtn").disabled = false;
      $("#fetchStopBtn").disabled = true;
      setPill(false, rc);
      loadDatasets();
    },
  });
}

async function runCommand() {
  const body = { command: CURRENT_CMD, values: collectValues() };
  const res = await fetch("/api/run", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  }).then((r) => r.json());

  if (!res.ok) {
    $("#formErrors").textContent = "⚠ " + (res.errors || ["failed to start"]).join(", ");
    return;
  }
  $("#formErrors").textContent = "";
  $("#console").innerHTML = "";
  setRunning(true);
  openStream({ target: "#console", onDone: (rc) => setRunning(false, rc) });
}

function openStream(opts) {
  streamTarget = opts.target;
  if (evtSource) evtSource.close();
  evtSource = new EventSource("/api/stream");

  evtSource.addEventListener("command", (e) => appendLine("$ " + e.data, "ok-line"));
  evtSource.addEventListener("line", (e) => appendLine(e.data));
  evtSource.addEventListener("done", (e) => {
    const rc = parseInt(e.data, 10);
    appendLine(`\n[process exited with code ${rc}]`, rc === 0 ? "ok-line" : "err-line");
    if (opts.onDone) opts.onDone(rc);
    evtSource.close();
    evtSource = null;
  });
  evtSource.onerror = () => {
    if (evtSource) { evtSource.close(); evtSource = null; }
  };
}

async function stopCommand() {
  await fetch("/api/stop", { method: "POST" });
}

function appendLine(text, cls) {
  const con = $(streamTarget);
  if (!con) return;
  let klass = cls;
  if (!klass) {
    const low = text.toLowerCase();
    if (low.includes("error") || low.includes("traceback") || low.includes("failed")) klass = "err-line";
    else if (low.includes("warning") || low.startsWith("warn")) klass = "warn-line";
  }
  con.append(el("div", { className: klass || "", textContent: text }));
  con.scrollTop = con.scrollHeight;
}

function setPill(running, rc) {
  const dot = $("#statusDot");
  const txt = $("#statusText");
  dot.className = "dot";
  if (running) { dot.classList.add("running"); txt.textContent = "running"; }
  else if (rc === undefined) { txt.textContent = "idle"; }
  else if (rc === 0) { dot.classList.add("done"); txt.textContent = "done"; }
  else { dot.classList.add("error"); txt.textContent = `exited (${rc})`; }
}

function setRunning(running, rc) {
  $("#runBtn").disabled = running;
  $("#stopBtn").disabled = !running;
  setPill(running, rc);
}

// ──────────────────────────────────────────────────────────────────────
// Tabs
// ──────────────────────────────────────────────────────────────────────
function setupTabs() {
  document.querySelectorAll(".tab").forEach((t) => {
    t.onclick = () => switchTab(t.dataset.tab);
  });
}

function switchTab(name) {
  document.querySelectorAll(".tab").forEach((x) => x.classList.toggle("active", x.dataset.tab === name));
  document.querySelectorAll(".tab-panel").forEach((x) => x.classList.toggle("active", x.id === "tab-" + name));
}

// ──────────────────────────────────────────────────────────────────────
// File editor
// ──────────────────────────────────────────────────────────────────────
async function loadFile() {
  const path = $("#fileSelect").value;
  if (!path) return;
  const res = await fetch("/api/file?path=" + encodeURIComponent(path)).then((r) => r.json());
  if (res.ok) {
    $("#fileEditor").value = res.content;
    msg("#fileMsg", "Loaded " + path, true);
  } else {
    msg("#fileMsg", res.error || "load failed", false);
  }
}

async function saveFile() {
  const path = $("#fileSelect").value;
  if (!path) return;
  const res = await fetch("/api/file", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ path, content: $("#fileEditor").value }),
  }).then((r) => r.json());
  msg("#fileMsg", res.ok ? "Saved " + path : res.error || "save failed", res.ok);
}

async function loadEnv() {
  const res = await fetch("/api/file?path=.env").then((r) => r.json());
  if (res.ok) $("#envEditor").value = res.content || envTemplate();
}

function envTemplate() {
  return [
    "BINANCE_API_KEY=",
    "BINANCE_API_SECRET=",
    "GOOGLE_API_KEY=",
    "OPENAI_API_KEY=",
    "TELEGRAM_BOT_TOKEN=",
    "TELEGRAM_CHAT_ID=",
    "",
  ].join("\n");
}

async function saveEnv() {
  const res = await fetch("/api/file", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ path: ".env", content: $("#envEditor").value }),
  }).then((r) => r.json());
  msg("#envMsg", res.ok ? "Saved .env" : res.error || "save failed", res.ok);
}

function msg(sel, text, ok) {
  const n = $(sel);
  n.textContent = text;
  n.classList.toggle("ok", !!ok);
}

init();
