"""Read-only localhost dashboard for a running DecompBench campaign (standard library only)."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import math
import os
from pathlib import Path
import re
import stat
import threading
import time
from urllib.parse import parse_qs, urlsplit

from src.reporting_policy import MODEL_EXCLUSIONS, model_exclusion, require_reportable_model
from src.campaign_plan import model_queue
from src.agent.policy_completion import TERMINAL_KIND, is_terminal_failure, TerminalCompletionError, approved_receipts, load_policy_summary, policy_is_complete


APP_ID = "decompbench-dashboard-v1"
ASSETS = Path(__file__).resolve().parent / "dashboard"
MODEL_NAMES = {"gpt-5.6-luna": "GPT-5.6 Luna", "gpt-5.6-sol": "GPT-5.6 Sol", "gpt-6-luna": "GPT-6 Luna", "gpt-6-sol": "GPT-6 Sol", "gpt-6.1-sol": "GPT-6.1 Sol", "gpt-6-astra": "GPT-6 Astra", "gpt-oss-120b-medium": "GPT-OSS 120B", "gemini-3.8-flash-medium": "Gemini 3.8 Flash", "gemini-3.1-pro-high": "Gemini 3.1 Pro", "claude-sonnet-4-6": "Claude Sonnet 4.6", "claude-opus-4-6-thinking": "Claude Opus 4.6"}
MODEL_NAMES.update({"claude-opus-5-5": "Claude Opus 5.5", "claude-sonnet-5-5": "Claude Sonnet 5.5"})
FAMILIES = {"ufl": "Facility location", "fctp": "Fixed-charge transportation", "km": "K-median", "lrp": "Location routing", "tsso": "Two-stage stochastic optimization", "bp": "Bin packing", "csp": "Cutting stock", "gap": "Generalized assignment", "shift": "Shift scheduling"}


def read_text(path: Path, limit: int = 4 * 1024 * 1024) -> str:
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
            raise ValueError("File is not a bounded regular file")
        data = stream.read(limit + 1)
        if len(data) > limit:
            raise ValueError("File exceeds its size limit")
        return data.decode("utf-8")


def redact(value: str) -> str:
    value = re.sub(r"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+", "[token redacted]", value)
    value = re.sub(r"(?im)((?:WLSACCESSID|WLSSECRET|WLSTOKEN|OPENAI_API_KEY|access_token|refresh_token|id_token)[\"']?\s*[:=]\s*)[^\s,;]+", r"\1[redacted]", value)
    value = re.sub(r"(?i)(academic license\s+)\d+", r"\1[redacted]", value)
    return re.sub(r"[A-Za-z0-9_.+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}", "[email redacted]", value)


def number(value):
    try:
        parsed = float(value)
        return parsed if math.isfinite(parsed) else None
    except (TypeError, ValueError):
        return None


def age(timestamp: str | None):
    if not timestamp:
        return None
    try:
        stamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        return max(0.0, time.time() - stamp.timestamp())
    except (ValueError, TypeError):
        return None


def process_alive(pid):
    if not isinstance(pid, int) or pid <= 0:
        return None
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return None


class JsonCache:
    """Keep the last complete JSON record while the producer replaces a file."""
    def __init__(self):
        self.values = {}

    def read(self, path: Path, default=None):
        try:
            info = path.stat()
            key = (info.st_mtime_ns, info.st_size)
            prior = self.values.get(path)
            if prior and prior[0] == key:
                return prior[1]
            value = json.loads(read_text(path))
            self.values[path] = (key, value)
            return value
        except (OSError, ValueError, UnicodeError):
            return self.values[path][1] if path in self.values else default


def first_attempt(summary):
    if not isinstance(summary, dict):
        return None
    attempts = summary.get("attempts", [])
    return next((item for item in attempts if item.get("attempt") == 1), None)


class Dashboard:
    def __init__(self, root: Path, historical_root: Path | None = None):
        self.root = root.resolve()
        self.historical_root = historical_root
        self.cache = JsonCache()
        self.lock = threading.RLock()
        self.last_snapshot = None
        self.last_built = 0.0

    def historical(self):
        if self.historical_root is None:
            return None
        data = self.cache.read(self.historical_root / "attempt_curves.json", {})
        cohorts = []
        for curve in data.get("attempt_curves", []):
            point = next((item for item in curve.get("points", []) if item.get("budget") == 1), None)
            if point:
                cohorts.append({"mode": curve["mode"], "instances": curve["num_instances"], "passed": point["counts"]["quality"], "rate": point["quality"], "execution": point["counts"]["execution"], "verified": point["counts"]["verification"]})
        return {"model": "gpt-5.6-luna", "name": "GPT-5.6 Luna", "cohorts": cohorts, "instances": data.get("coverage", {}).get("expected_data_instances", 81), "trajectories": data.get("coverage", {}).get("completed_instance_runs", 0), "note": "Saved first-attempt results from two historical cohorts. Original effort, judge and isolation settings are retained; these are excluded from the new-run totals."}

    def manifest(self):
        result = self.cache.read(self.root / "campaign.json")
        if not isinstance(result, dict):
            raise ValueError("Campaign manifest is not available")
        return result

    def case_directory(self, model, instance):
        require_reportable_model(model)
        manifest = self.manifest()
        if model not in [item["model"] for item in model_queue(manifest)]:
            raise ValueError("Unknown model")
        case = next((item for item in manifest["instances"] if item["instance"] == instance), None)
        if case is None:
            raise ValueError("Unknown instance")
        path = self.root / "runs" / model / "plain" / case["suite"] / case["problem"] / instance
        if not path.resolve().is_relative_to(self.root / "runs"):
            raise ValueError("Invalid result path")
        return path

    def details(self, model, instance):
        with self.lock:
            directory = self.case_directory(model, instance)
            summary = self.cache.read(directory / "instance_summary.json", {})
            attempt = first_attempt(summary)
            if attempt is None:
                raise ValueError("This case has no completed result yet")
            files = {}
            for name in ("solver.py", "alg_verifier.md", "exec.log"):
                path = directory / "attempts/attempt_01" / name
                if not path.resolve().is_relative_to(directory.resolve()):
                    continue
                try:
                    files[name] = redact(read_text(path, 1024 * 1024))
                except (OSError, ValueError, UnicodeError):
                    files[name] = "This artifact is not available."
            policies = {}
            if self.manifest().get("backend") == "native":
                approvals = approved_receipts(self.root)
                for mode in ("plain", "reflection"):
                    policy_directory = self.root / "runs" / model / mode / directory.parent.parent.name / directory.parent.name / instance
                    policy = load_policy_summary(policy_directory, self.cache.read(policy_directory / "instance_summary.json", {}), campaign_root=self.root, approvals=approvals)
                    policies[mode] = {key: policy.get(key) for key in ("complete", "completion_kind", "outcome", "terminal_reason", "actual_attempts", "unexecuted_attempt_slots")}
            return {"model": model, "model_name": MODEL_NAMES.get(model, model), "instance": instance, "status": attempt["status"], "objective": number(attempt.get("objective")), "reference": number(attempt.get("gold_objective", summary.get("gold_objective"))), "gap": number(attempt.get("gap_to_gold")), "execution_ok": bool(attempt.get("exec_ok")), "feasibility_ok": bool(attempt.get("feasibility_ok")), "algorithm_verdict": attempt.get("llm_verdict"), "issues": [redact(str(item)) for item in attempt.get("issues", [])], "files": files, "metrics": self.metric_fields(attempt), "policies": policies}

    @staticmethod
    def metric_fields(attempt):
        metrics = attempt.get("metrics", {})
        generation = metrics.get("generation", {})
        judge = metrics.get("algorithm_verification", {})
        solver = metrics.get("solver", {})
        return {"generation_seconds": number(generation.get("elapsed_seconds")), "judge_seconds": number(judge.get("elapsed_seconds")), "solver_seconds": number(metrics.get("execution_seconds")), "license_wait_seconds": number(solver.get("license_wait_seconds")), "generation_usage": generation.get("usage", {}), "judge_usage": judge.get("usage", {})}

    def snapshot(self, force=False):
        with self.lock:
            if not force and self.last_snapshot and time.monotonic() - self.last_built < 2:
                return self.last_snapshot
            manifest = self.manifest()
            native = manifest.get("backend") == "native"
            status = self.cache.read(self.root / "status.json", {})
            monitor = self.cache.read(self.root / "monitor/status.json", {})
            if native:
                from src.agent.resource_monitor import thermal_warnings
                monitor = dict(monitor)
                monitor.update(sampled_at=monitor.get("sampled_at_utc", monitor.get("finished_at_utc")), docker_and_controller_cpu_percent=monitor.get("cpu_percent"), system_free_memory_percent=monitor.get("system_free_percent"), thermal_warnings=thermal_warnings(monitor.get("thermal", "")))
            heartbeat_age = age(monitor.get("sampled_at"))
            alive = process_alive(status.get("pid"))
            effective_state = status.get("state", "not_started")
            pause = self.cache.read(self.root / "pause_request.json", {})
            if effective_state == "running" and pause.get("state") == "draining":
                effective_state = "draining_for_pause"
            if effective_state == "running":
                if alive is False:
                    effective_state = "supervisor_offline"
                elif heartbeat_age is None or heartbeat_age > 90:
                    effective_state = "heartbeat_stale"
                elif monitor.get("state") == "cooling":
                    effective_state = "cooling"
            models = []
            results = []
            expected = {case["instance"]: case for case in manifest["instances"]}
            raw_active = status.get("active_instances", {})
            if not raw_active and status.get("active_instance") and status.get("state") == "running":
                raw_active = {"experiment_0": {"instance": status["active_instance"], "started_at": status.get("case_started_at")}}
            active = []
            totals_usage = {"generation": {"input_tokens": 0, "output_tokens": 0, "cached_input_tokens": 0}, "judge": {"input_tokens": 0, "output_tokens": 0, "cached_input_tokens": 0}}
            agy_order = status.get("execution_agy_order", manifest.get("agy_order", []))
            if sorted(agy_order) != sorted(manifest.get("agy_order", [])):
                agy_order = manifest.get("agy_order", [])
            queue = model_queue(manifest if "model_queue" in manifest else dict(manifest, agy_order=agy_order))
            for specification in queue:
                model = specification["model"]
                if model_exclusion(model):
                    continue
                backend = {"codex-cli": "Codex", "agy-cli": "AGY", "claude-code-cli": "Claude Code"}[specification["backend"]]
                completed = []
                model_root = self.root / "runs" / model / "plain"
                for path in model_root.glob("*/*/*/instance_summary.json"):
                    instance = path.parent.name
                    if instance not in expected or not path.resolve().is_relative_to(self.root / "runs"):
                        continue
                    summary = self.cache.read(path, {})
                    attempt = first_attempt(summary)
                    if attempt is None:
                        continue
                    if (summary.get("model"), summary.get("instance")) != (model, instance):
                        continue
                    case = expected[instance]
                    if path != model_root / case["suite"] / case["problem"] / instance / "instance_summary.json":
                        continue
                    try:
                        modified = path.stat().st_mtime
                    except OSError:
                        continue
                    metrics = self.metric_fields(attempt)
                    row = {"model": model, "model_name": MODEL_NAMES.get(model, model), "instance": instance, "problem": case["problem"], "suite": case["suite"], "size": case["size"], "seed": case["seed"], "status": attempt["status"], "passed": attempt["status"] == "ok", "execution_ok": bool(attempt.get("exec_ok")), "feasibility_ok": bool(attempt.get("feasibility_ok")), "algorithm_ok": attempt.get("llm_verdict") == "pass", "objective": number(attempt.get("objective")), "reference": number(attempt.get("gold_objective", summary.get("gold_objective"))), "gap": number(attempt.get("gap_to_gold")), "solver_seconds": metrics["solver_seconds"], "finished_at": datetime.fromtimestamp(modified, timezone.utc).isoformat(), "metrics": metrics}
                    completed.append(row)
                    results.append(row)
                    for role, usage in (("generation", metrics["generation_usage"]), ("judge", metrics["judge_usage"])):
                        for key in totals_usage[role]:
                            value = usage.get(key, 0)
                            if isinstance(value, int):
                                totals_usage[role][key] += value
                done = {row["instance"] for row in completed}
                for worker, entry in raw_active.items():
                    if entry.get("model", status.get("active_model")) != model or (entry.get("instance") in done and not native) or entry.get("instance") not in expected:
                        continue
                    directory = self.case_directory(model, entry["instance"]) / "attempts/attempt_01"
                    if (directory / "llm_verifier_metrics.json").exists():
                        phase = "Saving result"
                    elif (directory / "solver_invocation.json").exists():
                        phase = "Algorithm review / Codex queue"
                    elif (directory / "agent_metrics.json").exists():
                        phase = "Solver execution / WLS wait"
                    else:
                        phase = "Generation / Codex queue" if backend == "Codex" else "Generation"
                    if native:
                        phase = f"{entry.get('mode', 'shared')} · attempt {entry.get('attempt', 1)}/{manifest['max_attempts']} · {entry.get('phase', 'starting')}"
                    active.append({"worker": worker, "model": model, "model_name": MODEL_NAMES.get(model, model), "instance": entry["instance"], "phase": phase, "started_at": entry.get("started_at"), "elapsed_seconds": age(entry.get("started_at")), "supervisor_active": effective_state in {"running", "cooling", "draining_for_pause"}})
                count = len(completed)
                if count == len(expected):
                    model_state = "complete"
                elif any(item["model"] == model for item in active):
                    model_state = "running" if effective_state == "running" else effective_state
                elif model == status.get("active_model") and effective_state not in {"running", "cooling"}:
                    model_state = effective_state
                elif backend == "AGY" and status.get("agy_state") == "pending_production_adapter":
                    model_state = "awaiting_setup"
                else:
                    model_state = "queued" if count == 0 else "in_progress"
                passed = sum(row["passed"] for row in completed)
                effort = specification["effort"].title() if specification.get("effort") not in (None, "catalog-slug") else ("Thinking" if model.startswith("claude") else ("High" if model.endswith("pro-high") else "Medium"))
                models.append({"id": model, "name": MODEL_NAMES.get(model, model), "backend": backend, "effort": effort, "state": model_state, "expected": len(expected), "completed": count, "remaining": len(expected) - count, "passed": passed, "failed": count - passed, "pass_rate": passed / count if count else None, "execution_passes": sum(row["execution_ok"] for row in completed), "feasibility_passes": sum(row["feasibility_ok"] for row in completed), "algorithm_passes": sum(row["algorithm_ok"] for row in completed)})
            results.sort(key=lambda row: row["finished_at"], reverse=True)
            total = sum(row["expected"] for row in models)
            passed = sum(row["passed"] for row in results)
            alerts = []
            if effective_state in {"infrastructure_stopped", "resource_stopped", "supervisor_offline", "heartbeat_stale"}:
                alerts.append({"kind": "error", "text": redact(status.get("error") or "The supervisor is stopped or its heartbeat is stale. Saved results remain available.")})
            if monitor.get("reason"):
                alerts.append({"kind": "warning", "text": str(monitor["reason"])})
            snapshot = {"app_id": APP_ID, "campaign": self.root.name, "generated_at": datetime.now(timezone.utc).isoformat(), "refresh_seconds": 5, "state": effective_state, "controller_state": status.get("state"), "process_alive": alive, "started_at": manifest.get("created_at"), "heartbeat_age_seconds": heartbeat_age, "last_activity_at": status.get("updated_at"), "totals": {"planned": total, "completed": len(results), "remaining": total - len(results), "passed": passed, "failed": len(results) - passed, "pass_rate": passed / len(results) if results else None, "active": sum(item["supervisor_active"] for item in active)}, "models": models, "active_workers": active, "results": results, "historical": self.historical(), "monitor": monitor, "alerts": alerts, "usage": totals_usage, "protocol": {"instances": len(expected), "families": len({case['problem'] for case in expected.values()}), "attempts": manifest.get("max_attempts", 1), "judge": MODEL_NAMES.get(manifest.get("judge_model"), manifest.get("judge_model")), "judge_effort": manifest.get("judge_effort"), "workers": manifest.get("workers", 2), "solver_timeout": manifest.get("solver_timeout", 60), "cpu_per_worker": manifest.get("cpu_limit_per_container", 1), "memory_per_worker": "2 GiB", "codex_concurrency": manifest.get("active_codex_invocation_limit", 1), "wls_reservations": manifest.get("max_wls_reservations", 2), "resource_revision": manifest.get("resource_revision", 1), "controller_revision": manifest.get("controller_revision", 1), "recovery_generation_allowance": manifest.get("infrastructure_recovery", {}).get("additional_generation_allowance", 0)}, "families": FAMILIES}
            if native:
                try:
                    terminal_approvals = approved_receipts(self.root)
                except TerminalCompletionError as error:
                    terminal_approvals = {}
                    alerts.append({"kind": "error", "text": redact(str(error))})
                result_index = {(row["model"], row["instance"]): row for row in results}
                consumed = set()
                totals_usage = {role: {key: 0 for key in ("input_tokens", "output_tokens", "cached_input_tokens")} for role in ("generation", "judge", "reflection")}
                for model in models:
                    completed_pairs = 0
                    policy_finished = {"plain": 0, "reflection": 0}
                    terminal_failures = {"plain": 0, "reflection": 0}
                    for case in expected.values():
                        both = True
                        for mode in ("plain", "reflection"):
                            directory = self.root / "runs" / model["id"] / mode / case["suite"] / case["problem"] / case["instance"]
                            summary = self.cache.read(directory / "instance_summary.json", {})
                            try:
                                summary = load_policy_summary(directory, summary, campaign_root=self.root, approvals=terminal_approvals)
                                finished = policy_is_complete(summary, manifest["max_attempts"], manifest.get("protocol_id"))
                            except TerminalCompletionError as error:
                                finished = False
                                alerts.append({"kind": "error", "text": redact(f"{model['id']} {case['instance']}: {error}")})
                            terminal = finished and is_terminal_failure(summary)
                            terminal_failures[mode] += bool(terminal)
                            both = both and finished
                            policy_finished[mode] += bool(finished)
                            row = result_index.get((model["id"], case["instance"]))
                            if row is not None:
                                row[mode + "_complete"] = bool(finished)
                                row[mode + "_outcome"] = ("terminal_failed" if terminal else ("passed" if any(a.get("status") == "ok" for a in summary.get("attempts", [])) else "failed")) if finished else "pending"
                                row[mode + "_terminal_reason"] = summary.get("terminal_reason") if terminal else None
                            for attempt in summary.get("attempts", []):
                                for key, role in (("generation", "generation"), ("algorithm_verification", "judge"), ("reflection", "reflection")):
                                    metric = attempt.get("metrics", {}).get(key, {})
                                    invocation = metric.get("invocation_id")
                                    if not invocation or invocation in consumed:
                                        continue
                                    consumed.add(invocation)
                                    for field in totals_usage[role]:
                                        value = metric.get("usage", {}).get(field, 0)
                                        if isinstance(value, int):
                                            totals_usage[role][field] += value
                        completed_pairs += both
                    model.update(completed_pairs=completed_pairs, policy_finished=policy_finished, policy_terminal_failures=terminal_failures, remaining_pairs=len(expected) - completed_pairs)
                    if completed_pairs == len(expected):
                        model["state"] = "complete"
                    elif any(item["model"] == model["id"] for item in active):
                        model["state"] = effective_state
                    elif model["state"] == "complete":
                        model["state"] = "in_progress"
                pairs = sum(m["completed_pairs"] for m in models)
                snapshot["totals"].update(first_attempt_completed=len(results), completed=pairs, remaining=total - pairs)
                snapshot["totals"]["terminal_policy_failures"] = sum(sum(model["policy_terminal_failures"].values()) for model in models)
                snapshot["usage"] = totals_usage
                snapshot["usage_note"] = "Committed attempt metrics, deduplicated by physical invocation ID; uncommitted and failed infrastructure calls are excluded."
                snapshot["protocol"].update(backend="native", shared_first_attempt=True, stop_after_success=True, wls_reservations=0, sample_interval=5, memory_scope="entire experiment", max_cpu_percent=250, max_rss_mib=2048)
                snapshot["protocol"]["codex_concurrency"] = status.get("active_codex_invocation_limit", snapshot["protocol"]["codex_concurrency"])
                snapshot["protocol"]["execution_phase_id"] = status.get("execution_phase_id")
                if effective_state in {"recovering", "draining_for_recovery"}:
                    alerts.append({"kind": "warning", "text": redact(status.get("error", "Waiting to resume the saved stage after a recovery pause."))})
            snapshot["reporting_exclusions"] = [dict(value) for model, value in MODEL_EXCLUSIONS.items() if model in [item["model"] for item in queue]]
            if manifest.get("schema_version") == 3:
                snapshot["historical_reference"] = manifest.get("historical_reuse")
            self.last_snapshot, self.last_built = snapshot, time.monotonic()
            return snapshot


def export_csv(snapshot):
    out = io.StringIO()
    columns = ["model", "instance", "problem", "size", "seed", "status", "execution_ok", "feasibility_ok", "algorithm_ok", "objective", "reference", "gap", "solver_seconds", "finished_at", "plain_complete", "reflection_complete", "plain_outcome", "reflection_outcome", "plain_terminal_reason", "reflection_terminal_reason"]
    writer = csv.DictWriter(out, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    for row in snapshot["results"]:
        if model_exclusion(row.get("model")):
            continue
        clean = {key: ("'" + value if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")) else value) for key, value in row.items()}
        writer.writerow(clean)
    return out.getvalue().encode("utf-8")


def handler_for(dashboard):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send_body(self, code, body, content_type, *, download=False):
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self' data:; object-src 'none'; base-uri 'none'; frame-ancestors 'none'")
            if download:
                self.send_header("Content-Disposition", 'attachment; filename="decompbench-results.csv"')
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def json_response(self, value, code=200):
            self.send_body(code, json.dumps(value, allow_nan=False).encode(), "application/json; charset=utf-8")

        def do_GET(self):
            allowed_hosts = {f"localhost:{self.server.server_port}", f"127.0.0.1:{self.server.server_port}"}
            if self.headers.get("Host", "").lower() not in allowed_hosts:
                self.json_response({"error": "Local access only"}, 403)
                return
            url = urlsplit(self.path)
            assets = {"/": ("index.html", "text/html; charset=utf-8"), "/app.js": ("app.js", "text/javascript; charset=utf-8"), "/style.css": ("style.css", "text/css; charset=utf-8"), "/favicon.svg": ("favicon.svg", "image/svg+xml")}
            try:
                if url.path in assets:
                    name, mime = assets[url.path]
                    self.send_body(200, (ASSETS / name).read_bytes(), mime)
                elif url.path == "/api/status":
                    self.json_response(dashboard.snapshot())
                elif url.path == "/api/case":
                    query = parse_qs(url.query)
                    self.json_response(dashboard.details(query.get("model", [""])[0], query.get("instance", [""])[0]))
                elif url.path == "/api/results.csv":
                    self.send_body(200, export_csv(dashboard.snapshot()), "text/csv; charset=utf-8", download=True)
                else:
                    self.json_response({"error": "Not found"}, 404)
            except (ValueError, OSError, KeyError) as error:
                self.json_response({"error": redact(str(error))[:300]}, 404 if url.path == "/api/case" else 503)
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/experiments/fixed81_20261007"))
    parser.add_argument("--historical-root", type=Path, default=Path(__file__).resolve().parents[2] / "DecompBench/results/luna_60s")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    dashboard = Dashboard(args.root, args.historical_root)
    dashboard.snapshot()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler_for(dashboard))
    print(json.dumps({"app_id": APP_ID, "url": f"http://localhost:{server.server_port}", "pid": os.getpid(), "campaign": str(args.root.resolve())}), flush=True)
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
