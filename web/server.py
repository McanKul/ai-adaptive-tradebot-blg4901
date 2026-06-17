#!/usr/bin/env python3
"""
web/server.py
=============
A local web control panel for the AI-Adaptive Trading Bot.

This is a thin GUI over the single entrypoint (``app.py``).  It does **not**
re-implement any trading logic — it introspects ``app.py``'s argparse
definitions to discover every setting of every subcommand
(``backtest``, ``walk-forward``, ``live``, ``dry-run``, ``sweep``,
``validate``), renders them as a web form, and then runs the exact same
``python app.py <command> ...`` invocation as a subprocess, streaming its
output back to the browser in real time.

Because the form is generated *from* ``app.py``, any flag added to the CLI
shows up in the UI automatically — there is no separate setting list to keep
in sync.

Run it:

    pip install flask          # only extra dependency
    python web/server.py       # then open http://127.0.0.1:8000

Everything runs locally; nothing is sent anywhere.
"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import threading

# ── Make the project root importable so we can introspect app.py ─────────
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from flask import Flask, Response, jsonify, request, send_from_directory  # noqa: E402

import app as app_module  # the single entrypoint  # noqa: E402

app = Flask(__name__, static_folder="static", template_folder="templates")

# ── Presentation-only grouping of fields (does not affect behaviour) ─────
# Maps an argparse ``dest`` to a section heading in the UI.  Anything not
# listed falls back to "General".
FIELD_GROUPS = {
    # Strategy & market
    "strategy": "Strategy & Market", "composite_spec": "Strategy & Market",
    "symbol": "Strategy & Market", "timeframe": "Strategy & Market",
    "strategy_params": "Strategy & Market",
    # Sizing & capital
    "margin_usd": "Sizing & Capital", "notional_usd": "Sizing & Capital",
    "qty": "Sizing & Capital", "leverage": "Sizing & Capital",
    "leverage_mode": "Sizing & Capital", "capital": "Sizing & Capital",
    # Exit
    "tp_pct": "Exit Rules", "sl_pct": "Exit Rules", "tick_exit": "Exit Rules",
    # Data & realism
    "data_dir": "Data & Realism", "realism_config": "Data & Realism",
    "export_trades": "Data & Realism",
    # Partial fills
    "enable_partial_fills": "Partial Fills", "liquidity_scale": "Partial Fills",
    "min_fill_ratio": "Partial Fills",
    # Cross-validation
    "cv_method": "Cross-Validation", "cv_n_splits": "Cross-Validation",
    "cv_embargo_pct": "Cross-Validation", "cv_train_pct": "Cross-Validation",
    "cv_n_test_splits": "Cross-Validation", "cv_aggregate": "Cross-Validation",
    "cv_expanding": "Cross-Validation", "cv_hyperband": "Cross-Validation",
    "cv_halving_factor": "Cross-Validation", "cv_min_active": "Cross-Validation",
    # walk-forward aliases (same dests via different option strings)
    # Sweep
    "param_grid": "Sweep", "csv_output": "Sweep", "top_n": "Sweep",
    "scorer_min_trades": "Sweep Filters", "min_trades": "Sweep Filters",
    "min_sharpe": "Sweep Filters", "max_dd": "Sweep Filters",
    "min_win_rate": "Sweep Filters", "cv_stability_weight": "Sweep Filters",
    "max_cv_std": "Sweep Filters",
    # Live / dry-run
    "config": "Live / Dry-run", "run_id": "Live / Dry-run",
    "sentiment": "Live / Dry-run", "force_live": "Live / Dry-run",
    "max_stamp_age_days": "Live / Dry-run",
    # Validate
    "real_money": "Validation",
}

# dests that point at an existing config-style file → offer a file picker
FILE_FIELDS = {"config", "composite_spec", "realism_config", "param_grid"}


def _build_parser() -> argparse.ArgumentParser:
    """Rebuild app.py's full parser using its own subparser builders."""
    parser = argparse.ArgumentParser(prog="app")
    sub = parser.add_subparsers(dest="command")
    app_module._add_backtest_parser(sub)
    app_module._add_walk_forward_parser(sub)
    app_module._add_live_parser(sub)
    app_module._add_dry_run_parser(sub)
    app_module._add_sweep_parser(sub)
    app_module._add_validate_parser(sub)
    return parser, sub


def build_schema() -> list[dict]:
    """Introspect every subcommand and return a JSON-serialisable schema.

    Each command -> list of parameters with their type, default, choices,
    required flag, help text, and UI hints.  This is the single source of
    truth the frontend renders, derived directly from ``app.py``.
    """
    _, sub = _build_parser()
    commands = []
    for name, p in sub.choices.items():
        params: dict[str, dict] = {}
        order: list[str] = []
        for a in p._actions:
            if isinstance(a, argparse._HelpAction):
                continue
            dest = a.dest
            if dest not in params:
                params[dest] = {
                    "dest": dest,
                    "help": (a.help or "").replace("%%", "%"),
                    "required": bool(a.required),
                    "choices": list(a.choices) if a.choices else None,
                    "default": a.default,
                    "group": FIELD_GROUPS.get(dest, "General"),
                }
                order.append(dest)
            entry = params[dest]
            if isinstance(a, argparse._StoreTrueAction):
                entry["type"] = "bool"
                entry.setdefault("flags", {})["true"] = a.option_strings[0]
            elif isinstance(a, argparse._StoreFalseAction):
                entry["type"] = "bool"
                entry.setdefault("flags", {})["false"] = a.option_strings[0]
            else:
                if a.type is float:
                    entry["type"] = "float"
                elif a.type is int:
                    entry["type"] = "int"
                elif a.choices:
                    entry["type"] = "choice"
                else:
                    entry["type"] = "str"
                entry["option"] = a.option_strings[0]

        ordered = [params[d] for d in order]
        for entry in ordered:
            if entry["dest"] in FILE_FIELDS:
                entry["widget"] = "file"
            elif entry["dest"] == "strategy":
                entry["widget"] = "strategy"
        commands.append({"name": name, "params": ordered})
    return commands


def build_argv(command: str, values: dict) -> list[str]:
    """Translate UI form ``values`` into a ``python app.py`` argv list.

    The same translation backs both the live "command preview" and the
    actual run, so what the user sees is exactly what executes.
    """
    schema = {c["name"]: c for c in build_schema()}
    if command not in schema:
        raise ValueError(f"Unknown command: {command}")

    argv = [command]
    for param in schema[command]["params"]:
        dest = param["dest"]
        ptype = param["type"]
        raw = values.get(dest, None)

        if ptype == "bool":
            flags = param.get("flags", {})
            # Tri-state for paired flags (e.g. --tick-exit / --no-tick-exit):
            #   "on" -> true flag, "off" -> false flag, anything else -> omit.
            # Single store_true flag: truthy -> emit the flag, else omit.
            if "true" in flags and "false" in flags:
                if raw == "on":
                    argv.append(flags["true"])
                elif raw == "off":
                    argv.append(flags["false"])
            else:
                if raw is True or raw == "on" or raw == "true":
                    argv.append(flags.get("true") or param.get("option"))
            continue

        # Value-bearing args: skip blanks (lets the CLI default apply).
        if raw is None:
            continue
        sval = str(raw).strip()
        if sval == "":
            continue
        argv.extend([param["option"], sval])

    return argv


def validate_required(command: str, values: dict) -> list[str]:
    """Return a list of human-readable errors for missing required fields."""
    schema = {c["name"]: c for c in build_schema()}
    errors = []
    for param in schema.get(command, {}).get("params", []):
        if param["required"]:
            raw = values.get(param["dest"], None)
            if raw is None or str(raw).strip() == "":
                opt = param.get("option", param["dest"])
                errors.append(f"{opt} is required")
    return errors


# ── Subprocess run management (single run at a time) ─────────────────────


class Run:
    """A single ``python app.py ...`` subprocess with a streamed log buffer."""

    def __init__(self, argv: list[str], command_str: str):
        self.argv = argv
        self.command_str = command_str
        self.lines: list[str] = []
        self.cond = threading.Condition()
        self.done = False
        self.returncode: int | None = None
        self.proc: subprocess.Popen | None = None

    def start(self) -> None:
        # start_new_session=True puts the child in its own process group so
        # Stop can terminate it (and any of its children) cleanly.
        self.proc = subprocess.Popen(
            self.argv,
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        threading.Thread(target=self._reader, daemon=True).start()

    def _append(self, line: str) -> None:
        with self.cond:
            self.lines.append(line)
            self.cond.notify_all()

    def _reader(self) -> None:
        assert self.proc and self.proc.stdout
        try:
            for line in self.proc.stdout:
                self._append(line.rstrip("\n"))
        finally:
            self.proc.wait()
            with self.cond:
                self.returncode = self.proc.returncode
                self.done = True
                self.cond.notify_all()

    def stop(self) -> bool:
        if self.proc and self.proc.poll() is None:
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
            except ProcessLookupError:
                return False
            return True
        return False

    def stream(self):
        """Yield SSE frames: full history first, then live lines, then done."""
        idx = 0
        while True:
            with self.cond:
                while idx >= len(self.lines) and not self.done:
                    self.cond.wait(timeout=1.0)
                pending = self.lines[idx:]
                idx = len(self.lines)
                finished = self.done and idx >= len(self.lines)
                rc = self.returncode
            for line in pending:
                yield _sse("line", line)
            if finished:
                yield _sse("done", str(rc))
                return


_current: Run | None = None
_run_lock = threading.Lock()


def _sse(event: str, data: str) -> str:
    # Split multi-line payloads into multiple data: fields per the SSE spec.
    body = "".join(f"data: {ln}\n" for ln in data.split("\n"))
    return f"event: {event}\n{body}\n"


# ── Path safety for the file editor ──────────────────────────────────────


def _safe_path(rel: str) -> str:
    """Resolve *rel* against the repo root, rejecting anything outside it."""
    full = os.path.realpath(os.path.join(REPO_ROOT, rel))
    root = os.path.realpath(REPO_ROOT)
    if full != root and not full.startswith(root + os.sep):
        raise ValueError("Path escapes the project directory")
    return full


# ── Routes ───────────────────────────────────────────────────────────────


@app.route("/")
def index():
    return send_from_directory("templates", "index.html")


@app.route("/api/schema")
def api_schema():
    return jsonify({"commands": build_schema()})


@app.route("/api/strategies")
def api_strategies():
    try:
        from core.bootstrap import register_defaults
        from core.factories.strategy_factory import StrategyFactory
        register_defaults()
        return jsonify({"strategies": StrategyFactory.list_available()})
    except Exception as e:  # never let a missing dep break the page
        return jsonify({"strategies": [], "error": str(e)})


@app.route("/api/files")
def api_files():
    """List config-style files (yaml/yml/json) under the repo, for pickers."""
    exts = (".yaml", ".yml", ".json")
    skip = {".git", "node_modules", ".venv", "venv", "__pycache__", "web"}
    found = []
    for dirpath, dirnames, filenames in os.walk(REPO_ROOT):
        dirnames[:] = [d for d in dirnames if d not in skip]
        for fn in filenames:
            if fn.endswith(exts):
                rel = os.path.relpath(os.path.join(dirpath, fn), REPO_ROOT)
                found.append(rel)
    found.sort()
    return jsonify({"files": found})


@app.route("/api/file", methods=["GET", "POST"])
def api_file():
    """Load (GET ?path=) or save (POST {path, content}) a text file."""
    if request.method == "GET":
        rel = request.args.get("path", "")
        try:
            full = _safe_path(rel)
            with open(full, "r", encoding="utf-8") as f:
                return jsonify({"ok": True, "path": rel, "content": f.read()})
        except FileNotFoundError:
            return jsonify({"ok": True, "path": rel, "content": ""})
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 400

    data = request.get_json(force=True) or {}
    rel = data.get("path", "")
    content = data.get("content", "")
    try:
        full = _safe_path(rel)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w", encoding="utf-8") as f:
            f.write(content)
        return jsonify({"ok": True, "path": rel})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400


@app.route("/api/datasets")
def api_datasets():
    """List downloaded tick datasets under data/ticks/<SYMBOL>/."""
    base = os.path.join(REPO_ROOT, "data", "ticks")
    out = []
    if os.path.isdir(base):
        for sym in sorted(os.listdir(base)):
            d = os.path.join(base, sym)
            if not os.path.isdir(d):
                continue
            days = sorted(f[:-4] for f in os.listdir(d) if f.endswith(".csv"))
            if not days:
                continue
            out.append({"symbol": sym, "days": len(days),
                        "start": days[0], "end": days[-1]})
    return jsonify({"datasets": out})


@app.route("/api/fetch", methods=["POST"])
def api_fetch():
    """Download tick data via tools/fetch_ticks.py (streamed like a run)."""
    global _current
    data = request.get_json(force=True) or {}
    symbol = (data.get("symbol") or "").strip().upper()
    start = (data.get("start") or "").strip()
    end = (data.get("end") or "").strip()
    market = (data.get("market_type") or "um").strip()
    dtype = (data.get("data_type") or "aggTrades").strip()

    errors = []
    if not symbol:
        errors.append("symbol is required")
    if not start:
        errors.append("start date is required")
    if not end:
        errors.append("end date is required")
    if errors:
        return jsonify({"ok": False, "errors": errors}), 400

    argv = ["tools/fetch_ticks.py", "--symbol", symbol, "--start", start,
            "--end", end, "--output", "data/ticks",
            "--market-type", market, "--data-type", dtype]
    with _run_lock:
        if _current and not _current.done:
            return jsonify({"ok": False, "errors": ["A run is already in progress. Stop it first."]}), 409
        full = [sys.executable, *argv]
        cmdstr = " ".join([os.path.basename(sys.executable), *_shell_join(argv)])
        run = Run(full, cmdstr)
        run.start()
        _current = run
    return jsonify({"ok": True, "command": cmdstr})


@app.route("/api/preview", methods=["POST"])
def api_preview():
    data = request.get_json(force=True) or {}
    command = data.get("command", "")
    values = data.get("values", {})
    try:
        argv = build_argv(command, values)
    except ValueError as e:
        return jsonify({"ok": False, "error": str(e)}), 400
    cmd = " ".join([os.path.basename(sys.executable), "app.py", *_shell_join(argv)])
    return jsonify({"ok": True, "command": cmd, "errors": validate_required(command, values)})


def _shell_join(argv: list[str]) -> list[str]:
    out = []
    for a in argv:
        out.append(f"'{a}'" if (" " in a or '"' in a or "{" in a) else a)
    return out


@app.route("/api/run", methods=["POST"])
def api_run():
    global _current
    data = request.get_json(force=True) or {}
    command = data.get("command", "")
    values = data.get("values", {})

    errors = validate_required(command, values)
    if errors:
        return jsonify({"ok": False, "errors": errors}), 400

    try:
        argv = build_argv(command, values)
    except ValueError as e:
        return jsonify({"ok": False, "errors": [str(e)]}), 400

    with _run_lock:
        if _current and not _current.done:
            return jsonify({"ok": False, "errors": ["A run is already in progress. Stop it first."]}), 409
        full_argv = [sys.executable, "app.py", *argv]
        command_str = " ".join([os.path.basename(sys.executable), "app.py", *_shell_join(argv)])
        run = Run(full_argv, command_str)
        run.start()
        _current = run

    return jsonify({"ok": True, "command": command_str})


@app.route("/api/stream")
def api_stream():
    run = _current
    if run is None:
        return jsonify({"ok": False, "error": "No run started yet."}), 404

    def gen():
        yield _sse("command", run.command_str)
        yield from run.stream()

    resp = Response(gen(), mimetype="text/event-stream")
    resp.headers["Cache-Control"] = "no-cache"
    resp.headers["X-Accel-Buffering"] = "no"
    return resp


@app.route("/api/stop", methods=["POST"])
def api_stop():
    run = _current
    if run is None:
        return jsonify({"ok": False, "error": "Nothing running."}), 404
    stopped = run.stop()
    return jsonify({"ok": stopped})


@app.route("/api/status")
def api_status():
    run = _current
    if run is None:
        return jsonify({"running": False})
    return jsonify({
        "running": not run.done,
        "command": run.command_str,
        "returncode": run.returncode,
        "lines": len(run.lines),
    })


def main(host: str | None = None, port: int | None = None) -> None:
    """Launch the control-panel web server (used by ``python app.py web``)."""
    host = host or os.environ.get("WEB_HOST", "127.0.0.1")
    port = int(port or os.environ.get("WEB_PORT", "8000"))
    print("\n  AI-Adaptive Trading Bot — Web Control Panel")
    print(f"  Open  http://{host}:{port}  in your browser\n")
    app.run(host=host, port=port, threaded=True, debug=False)


if __name__ == "__main__":
    main()
