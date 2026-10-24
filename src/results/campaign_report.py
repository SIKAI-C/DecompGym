"""Read-only, paired native/hybrid campaign auditing and physical-cost reporting."""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np

from src.experiments.verification import OutputVerifier
from src.experiments.completion import TerminalCompletionError, approved_receipts, is_terminal_failure, load_policy_summary, policy_is_complete
from src.results.policy import default_models, reported, run_identity


MODES = ("plain", "reflection")
SIZES = ("small", "medium", "large")
STAGES = {"agent": "generation", "llm_verifier": "algorithm_verification", "reflection": "reflection"}
OUTCOMES = ("execution", "verification", "quality")
BOOTSTRAP_SEED = 20261013
BOOTSTRAP_RESAMPLES = 4000


def now():
    return datetime.now(timezone.utc).isoformat()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


class Snapshot:
    """Cache one version of each input; detect replacement while the reader works."""
    def __init__(self, root):
        self.root = root.resolve()
        self.started_at = now()
        self.contents = {}
        self.signatures = {}
        self.errors = []

    def relative(self, path):
        return str(path.relative_to(self.root))

    @staticmethod
    def signature(path):
        stat = path.stat()
        return stat.st_ino, stat.st_size, stat.st_mtime_ns

    def bytes(self, path, required=True):
        path = Path(path)
        if path in self.contents:
            return self.contents[path]
        try:
            if path.is_symlink() or not path.resolve().is_relative_to(self.root):
                raise ValueError("Artifact is a symlink or leaves the campaign")
            before = self.signature(path)
            content = path.read_bytes()
            if before != self.signature(path):
                raise ValueError("Artifact changed during read")
            self.signatures[path] = before
            self.contents[path] = content
            return content
        except (OSError, ValueError) as error:
            if required or path.exists():
                self.errors.append({"path": self.relative(path), "issue": str(error)})
            return None

    def json(self, path, required=False):
        content = self.bytes(path, required)
        if content is None:
            return None
        try:
            value = json.loads(content)
            if not isinstance(value, dict):
                raise ValueError("Expected a JSON object")
            return value
        except (ValueError, RecursionError) as error:
            self.errors.append({"path": self.relative(path), "issue": str(error)})
            return None

    def hash_matches(self, path, expected):
        content = self.bytes(path)
        return content is not None and hashlib.sha256(content).hexdigest() == expected

    def finish(self):
        changed = []
        for path, before in self.signatures.items():
            try:
                if self.signature(path) != before:
                    changed.append(self.relative(path))
            except OSError:
                changed.append(self.relative(path))
        return {"started_at_utc": self.started_at, "finished_at_utc": now(), "files_read": len(self.contents),
                "changed_during_read": changed, "read_errors": self.errors,
                "consistency": "Per-file stable read; concurrently changing pairs are excluded. New files after directory enumeration appear in the next snapshot."}


def queue_for(manifest):
    if manifest.get("model_queue"):
        return manifest["model_queue"]
    return [{"model": model, "backend": backend} for key, backend in (("gpt_order", "codex-cli"), ("agy_order", "agy-cli"), ("claude_order", "claude-code-cli")) for model in manifest.get(key, [])]


def invocation_issues(invocation, backend, model, manifest):
    if not invocation:
        return ["Missing invocation evidence"]
    issues = []
    if invocation.get("backend") != backend:
        issues.append("Invocation backend differs")
    if invocation.get("requested_model", invocation.get("model")) != model:
        issues.append("Invocation requested model differs")
    if invocation.get("transport") == "docker":
        if not invocation.get("ephemeral") or invocation.get("external_context") is not False:
            issues.append("Missing fresh Docker context evidence")
        expected_image = manifest.get("execution_environment", {}).get("image_id")
        if expected_image and invocation.get("image_id") != expected_image:
            issues.append("Docker image differs from frozen environment")
        if not invocation.get("cleanup_verified"):
            issues.append("Docker cleanup was not verified")
    elif backend == "codex-cli":
        if not (invocation.get("ephemeral") and invocation.get("ignore_user_config") and invocation.get("project_doc_max_bytes") == 0):
            issues.append("Missing fresh Codex invocation evidence")
    elif backend in ("agy-cli", "claude-code-cli"):
        if not invocation.get("fresh_conversation") or invocation.get("resume") is not False or not invocation.get("sandbox"):
            issues.append("Missing fresh backend invocation evidence")
    else:
        issues.append("Unsupported backend")
    return issues


def metrics_equal(left, right):
    # Adapter-level metrics can precede the runner adding its physical ID.
    return {k: v for k, v in left.items() if k != "invocation_id"} == {k: v for k, v in right.items() if k != "invocation_id"}


def score_base(attempt):
    value = deepcopy(attempt)
    value.get("metrics", {}).pop("reflection", None)
    return value


def success(attempt, gold, threshold):
    return bool(attempt.get("verify_ok") and finite(attempt.get("objective")) and abs(attempt["objective"] - gold) / max(1, abs(gold)) <= threshold)


def curves(attempts, gold, threshold, budget):
    result = {}
    for k in range(1, budget + 1):
        prefix = [a for a in attempts if a["attempt"] <= k]
        result[str(k)] = {"execution": any(a.get("exec_ok") for a in prefix), "verification": any(a.get("verify_ok") for a in prefix), "quality": any(success(a, gold, threshold) for a in prefix)}
    return result


def frozen_output_checks(root, checks):
    """Invoke only the hash-audited frozen verifier; never import candidate solvers."""
    if not checks:
        return []
    # The paper's campaigns froze the older layout (src/agent/output_verifier.py); new campaigns freeze src/experiments/verification.py.
    script = """import json,sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
if (Path(sys.argv[1])/'src/agent/output_verifier.py').is_file():
 from src.agent.output_verifier import OutputVerifier
else:
 from src.experiments.verification import OutputVerifier
checks=json.load(sys.stdin)
results=[]
for c in checks:
 try:
  results.append(OutputVerifier.verify_output(c['suite'],c['problem'],Path(c['data']),Path(c['output']),1e-6))
 except Exception as error:
  results.append({'ok':False,'issues':['Verifier error: '+type(error).__name__+': '+str(error)]})
json.dump(results,sys.stdout)
"""
    result = subprocess.run([sys.executable, "-B", "-c", script, str(root / "controller")], input=json.dumps(checks), text=True, capture_output=True, timeout=180, check=True, cwd=root / "controller")
    return json.loads(result.stdout)


def audit_policy(snapshot, manifest, specification, case, mode, approvals, output_checks):
    root = snapshot.root
    model = specification["model"]
    directory = root / "runs" / model / mode / case["suite"] / case["problem"] / case["instance"]
    row = {"model": model, "mode": mode, **{k: case[k] for k in ("suite", "problem", "instance", "size", "seed")}, "path": snapshot.relative(directory), "state": "pending", "audit_issues": [], "attempts": [], "actual_attempts": 0, "reused_slots": 0, "unexecuted_attempt_slots": manifest["max_attempts"], "call_ids": [], "solver_ids": []}
    raw = snapshot.json(directory / "instance_summary.json")
    if raw is None:
        return row
    issues = row["audit_issues"]
    summary = raw
    for receipt_name in ("terminal_failure.json", "reflection_delivery_failure.json"):
        terminal_record = snapshot.json(directory / receipt_name)
        if terminal_record:
            for name in terminal_record.get("evidence_sha256", {}):
                snapshot.bytes(directory / name)
    try:
        summary = load_policy_summary(directory, raw, campaign_root=root, approvals=approvals)
    except TerminalCompletionError as error:
        issues.append(str(error))
    terminal = is_terminal_failure(summary)
    budget = manifest["max_attempts"]
    attempts = summary.get("attempts", [])
    row["attempts"] = attempts
    identity_record = snapshot.json(directory / "identity.json", True) or {}
    canonical = root / "dataset" / case["suite"] / case["problem"] / case["instance"]
    for key, expected in {"model": model, "instance": case["instance"], "suite": case["suite"], "problem": case["problem"], "reflection": mode == "reflection", "max_attempts": budget, "protocol_id": manifest["protocol_id"], "gap_threshold": manifest.get("gap_threshold", 0.01), "backend": specification["backend"]}.items():
        if summary.get(key) != expected:
            issues.append(f"Summary {key} differs")
    for key, expected in {"model": model, "mode": mode, "max_attempts": budget, "protocol_id": manifest["protocol_id"]}.items():
        if identity_record.get(key) != expected:
            issues.append(f"Identity {key} differs")
    for name in ("data.json", "prompt.txt"):
        expected = case["sha256"].get(name)
        if not expected or identity_record.get("inputs", {}).get(name) != expected or not snapshot.hash_matches(directory / name, expected):
            issues.append(f"Runtime input differs: {name}")
    for name in ("meta.json", "gold_monolithic.json"):
        if identity_record.get("references", {}).get(name) != case["sha256"].get(name):
            issues.append(f"Reference identity differs: {name}")
    gold = OutputVerifier.load_gold_objective(canonical, case["suite"])
    if not finite(gold):
        issues.append("Canonical reference objective unavailable")
        row["state"] = "audit_failed"
        return row
    row["gold_objective"] = gold
    if [a.get("attempt") for a in attempts] != list(range(1, len(attempts) + 1)) or len(attempts) > budget:
        issues.append("Attempt sequence is not a valid prefix")
    solved = None
    for attempt in attempts:
        index = attempt.get("attempt")
        if type(index) is not int or not 1 <= index <= budget:
            continue
        if attempt.get("reused_from_previous"):
            row["reused_slots"] += 1
            if solved is None or attempt.get("first_success_attempt") != solved["attempt"] or any(attempt.get(k) != solved.get(k) for k in ("objective", "gap_to_gold", "exec_ok", "verify_ok")):
                issues.append(f"Attempt {index}: invalid early-success reuse")
            continue
        row["actual_attempts"] += 1
        attempt_dir = directory / "attempts" / f"attempt_{index:02d}"
        score = snapshot.json(attempt_dir / "score_receipt.json", True) or {}
        saved = snapshot.json(attempt_dir / "summary.json", True)
        base = score_base(attempt)
        if score.get("result") != base or saved != base:
            issues.append(f"Attempt {index}: score and summary differ")
        for filename, digest in score.get("artifact_sha256", {}).items():
            if not snapshot.hash_matches(attempt_dir / filename, digest):
                issues.append(f"Attempt {index}: changed scored artifact {filename}")
        if not finite(attempt.get("gold_objective")) or not math.isclose(attempt["gold_objective"], gold, rel_tol=1e-10, abs_tol=1e-8):
            issues.append(f"Attempt {index}: reference objective differs")
        stages = attempt.get("metrics", {})
        if "generation" not in stages:
            issues.append(f"Attempt {index}: generation metrics missing")
        if stages.get("generation", {}).get("ok"):
            if "algorithm_verification" not in stages:
                issues.append(f"Attempt {index}: algorithm-judge metrics missing")
            execution = snapshot.json(attempt_dir / "execution.json", True) or {}
            for filename, key in (("solver.py", "solver_sha256"), ("output.json", "output_sha256")):
                if execution.get(key) and not snapshot.hash_matches(attempt_dir / filename, execution[key]):
                    issues.append(f"Attempt {index}: execution {filename} hash differs")
            if execution.get("exec_ok") != attempt.get("exec_ok") or execution.get("execution_seconds") != stages.get("execution_seconds"):
                issues.append(f"Attempt {index}: execution result or duration differs")
        if mode == "reflection" and not terminal and not success(attempt, gold, manifest.get("gap_threshold", 0.01)) and index < budget and (summary.get("complete") or index < len(attempts)) and "reflection" not in stages:
            issues.append(f"Attempt {index}: consumed reflection stage missing")
        if manifest.get("agent_execution") == "docker" and index == budget and "reflection" in stages:
            issues.append(f"Attempt {index}: unexpected final reflection under fresh protocol")
        for step, stage in STAGES.items():
            metrics = stages.get(stage)
            if metrics is None:
                continue
            receipt = snapshot.json(attempt_dir / f"{step}_receipt.json", True) or {}
            if receipt.get("result") != metrics:
                issues.append(f"Attempt {index}: {stage} receipt differs")
            call_id = metrics.get("invocation_id")
            if not call_id:
                issues.append(f"Attempt {index}: {stage} has no physical invocation ID")
            else:
                row["call_ids"].append(call_id)
            invocation = snapshot.json(attempt_dir / f"{step}_invocation.json", True) or {}
            expected_backend = "codex-cli" if step == "llm_verifier" else specification["backend"]
            expected_model = manifest["judge_model"] if step == "llm_verifier" else model
            issues.extend(f"Attempt {index}, {stage}: {issue}" for issue in invocation_issues(invocation, expected_backend, expected_model, manifest))
            target = {"agent": "solver.py", "llm_verifier": "alg_verifier.md", "reflection": "debug.md"}[step]
            if metrics.get("artifact_sha256") and not snapshot.hash_matches(attempt_dir / target, metrics["artifact_sha256"]):
                issues.append(f"Attempt {index}: changed {stage} delivery")
            if metrics.get("file_delivery_failure") and manifest.get("agent_execution") == "docker":
                content = snapshot.bytes(attempt_dir / "debug.md")
                if content is None or content.strip():
                    issues.append(f"Attempt {index}: reflection omission must retain empty debug.md")
                for name, digest in metrics.get("delivery_evidence_sha256", {}).items():
                    if not snapshot.hash_matches(attempt_dir / "calls" / call_id / name, digest):
                        issues.append(f"Attempt {index}: reflection omission evidence changed")
        if attempt.get("verify_ok"):
            if not attempt.get("exec_ok") or attempt.get("llm_verdict") != "pass":
                issues.append(f"Attempt {index}: verification lacks execution or accepted judgment")
            notes = snapshot.bytes(attempt_dir / "alg_verifier.md") or b""
            if notes.decode(errors="replace").strip().splitlines()[:1] != ["VERDICT: PASS"]:
                issues.append(f"Attempt {index}: saved judge verdict differs")
            output_checks.append({"row": row, "attempt": index, "objective": attempt.get("objective"), "suite": case["suite"], "problem": case["problem"], "data": str(canonical / "data.json"), "output": str(attempt_dir / "output.json")})
        if finite(attempt.get("objective")) and attempt.get("gap_to_gold") is not None:
            gap = abs(attempt["objective"] - gold) / max(1, abs(gold))
            if not finite(attempt["gap_to_gold"]) or not math.isclose(gap, attempt["gap_to_gold"], abs_tol=1e-8, rel_tol=1e-7):
                issues.append(f"Attempt {index}: objective gap differs")
        if success(attempt, gold, manifest.get("gap_threshold", 0.01)):
            solved = attempt
    try:
        complete = policy_is_complete(summary, budget, manifest["protocol_id"])
    except TerminalCompletionError as error:
        issues.append(str(error))
        complete = False
    if summary.get("complete") and not complete:
        issues.append("Claimed completion lacks full attempt slots or validated terminal receipt")
    row.update(state="complete" if complete else "partial", complete=complete, completion_kind=summary.get("completion_kind"), unexecuted_attempt_slots=budget - row["actual_attempts"] - row["reused_slots"], attempt_metrics=curves(attempts, gold, manifest.get("gap_threshold", 0.01), budget))
    if issues:
        row["state"] = "audit_failed"
    return row


def scan_costs(snapshot, manifest, pairs):
    calls = {}
    executions = {}
    conflicts = []
    for pair in pairs:
        for policy in pair["policies"].values():
            directory = snapshot.root / policy["path"]
            shared = snapshot.json(directory / "shared_first.json") or {}
            for attempt_dir in sorted((directory / "attempts").glob("attempt_*")):
                try:
                    index = int(attempt_dir.name.split("_")[-1])
                except ValueError:
                    continue
                receipts = {step: snapshot.json(attempt_dir / f"{step}_receipt.json") for step in STAGES}
                committed = {receipt["result"].get("invocation_id"): step for step, receipt in receipts.items() if receipt and receipt.get("result")}
                directories = sorted((attempt_dir / "calls").glob("*"))
                for call_dir in directories:
                    if not call_dir.is_dir():
                        continue
                    request = snapshot.json(call_dir / "request.json") or {}
                    step = request.get("step")
                    if step not in STAGES:
                        step = next((step for step in STAGES if (call_dir / f"{step}_metrics.json").exists()), None)
                    if step is None:
                        conflicts.append({"path": snapshot.relative(call_dir), "issue": "Call stage unavailable"})
                        continue
                    metrics = snapshot.json(call_dir / f"{step}_metrics.json")
                    invocation = snapshot.json(call_dir / f"{step}_invocation.json") or {}
                    container = snapshot.json(call_dir / f"{step}_container.json") or {}
                    physical_evidence = bool(metrics is not None or invocation or (container.get("container_id") and container.get("phase") in ("running", "stopped", "collected", "complete")))
                    call_id = call_dir.name
                    rejected = snapshot.json(call_dir / "rejected.json")
                    if call_id in committed and metrics is not None and not metrics_equal(metrics, receipts[committed[call_id]]["result"]):
                        conflicts.append({"path": snapshot.relative(call_dir), "issue": "Physical metrics differ from committed receipt"})
                    if call_id in committed:
                        classification = "shared_first" if index == 1 and step != "reflection" else "policy"
                    elif metrics is None:
                        classification = "in_flight_or_interrupted" if physical_evidence else "request_start_unconfirmed"
                    elif rejected or not metrics.get("ok"):
                        classification = "infrastructure_or_rejected"
                    else:
                        classification = "uncommitted_completed"
                    backend = invocation.get("backend", "codex-cli" if step == "llm_verifier" else pair["backend"])
                    usage = (metrics or {}).get("usage")
                    available = (metrics or {}).get("usage_available", invocation.get("usage_available", bool(usage)))
                    valid_usage = isinstance(usage, dict) and all(type(v) is int and v >= 0 for v in usage.values())
                    duration = (metrics or {}).get("elapsed_seconds")
                    entry = {"invocation_id": call_id, "model": pair["model"], "suite": pair["suite"], "problem": pair["problem"], "instance": pair["instance"], "size": pair["size"], "mode": "shared" if classification == "shared_first" else policy["mode"], "attempt": index, "stage": STAGES[step], "backend": backend, "requested_model": invocation.get("requested_model", invocation.get("model")), "classification": classification, "ok": (metrics or {}).get("ok"), "terminal_state": (metrics or {}).get("terminal_state"), "reflection_delivery_failure": (metrics or {}).get("file_delivery_failure", False), "usage_available": bool(available and valid_usage), "usage": usage if available and valid_usage else None, "elapsed_seconds": duration if finite(duration) and duration >= 0 else None, "created_at": invocation.get("created_at"), "path": snapshot.relative(call_dir), "evidence": "physical_call_directory", "physical_call_confirmed": physical_evidence}
                    if call_id in calls:
                        previous = calls[call_id]
                        if any(previous.get(k) != entry.get(k) for k in ("model", "instance", "stage", "usage", "elapsed_seconds")):
                            conflicts.append({"path": entry["path"], "issue": "Conflicting duplicate physical invocation ID"})
                    else:
                        calls[call_id] = entry
                # Top-level copies have no new cost. They can retain an old call whose directory is absent.
                for step, receipt in receipts.items():
                    if not receipt:
                        continue
                    metrics = receipt.get("result", {})
                    call_id = metrics.get("invocation_id")
                    if not call_id or call_id in calls:
                        continue
                    invocation = snapshot.json(attempt_dir / f"{step}_invocation.json") or {}
                    usage = metrics.get("usage")
                    available = invocation.get("usage_available", bool(usage))
                    calls[call_id] = {"invocation_id": call_id, "model": pair["model"], "suite": pair["suite"], "problem": pair["problem"], "instance": pair["instance"], "size": pair["size"], "mode": "shared" if index == 1 and step != "reflection" else policy["mode"], "attempt": index, "stage": STAGES[step], "backend": invocation.get("backend", "unknown"), "requested_model": invocation.get("requested_model", invocation.get("model")), "classification": "shared_first" if index == 1 and step != "reflection" else "policy", "ok": metrics.get("ok"), "terminal_state": metrics.get("terminal_state"), "reflection_delivery_failure": metrics.get("file_delivery_failure", False), "usage_available": bool(available), "usage": usage if available else None, "elapsed_seconds": metrics.get("elapsed_seconds"), "created_at": invocation.get("created_at"), "path": snapshot.relative(attempt_dir), "evidence": "receipt_only", "physical_call_confirmed": True}
                execution = snapshot.json(attempt_dir / "execution.json")
                if execution:
                    solver_id = (shared.get("id") or identity({"protocol": manifest["protocol_id"], "model": pair["model"], "instance": pair["instance"], "attempt": 1})) if index == 1 else snapshot.relative(attempt_dir)
                    entry = {"solver_id": solver_id, "model": pair["model"], "instance": pair["instance"], "mode": "shared" if index == 1 else policy["mode"], "attempt": index, "execution_seconds": execution.get("execution_seconds"), "solver_sha256": execution.get("solver_sha256"), "output_sha256": execution.get("output_sha256"), "path": snapshot.relative(attempt_dir)}
                    if solver_id in executions and any(executions[solver_id].get(k) != entry.get(k) for k in ("execution_seconds", "solver_sha256", "output_sha256")):
                        conflicts.append({"path": entry["path"], "issue": "Shared solver execution copies differ"})
                    else:
                        executions[solver_id] = entry
                    policy["solver_ids"].append(solver_id)
    return list(calls.values()), list(executions.values()), conflicts


def cost_summary(calls, executions=()):
    groups = defaultdict(list)
    for call in calls:
        if call.get("physical_call_confirmed", True):
            groups[(call["stage"], call["backend"])].append(call)
    stages = []
    for (stage, backend), rows in sorted(groups.items()):
        fields = sorted({key for row in rows for key in (row.get("usage") or {})})
        usage = {}
        for key in fields:
            values = [row["usage"][key] for row in rows if row.get("usage") is not None and key in row["usage"]]
            usage[key] = {"observed_sum": sum(values), "available_calls": len(values), "missing_calls": len(rows) - len(values), "complete": len(values) == len(rows)}
        durations = [row["elapsed_seconds"] for row in rows if finite(row.get("elapsed_seconds"))]
        stages.append({"stage": stage, "backend": backend, "calls": len(rows), "usage_available_calls": sum(row["usage_available"] for row in rows), "usage_missing_calls": sum(not row["usage_available"] for row in rows), "raw_usage_fields": usage, "elapsed_seconds_observed_sum": sum(durations) if durations else None, "elapsed_seconds_missing_calls": len(rows) - len(durations)})
    durations = [r["execution_seconds"] for r in executions if finite(r.get("execution_seconds"))]
    return {"recorded_requests": len(calls), "physical_cli_calls": sum(c.get("physical_call_confirmed", True) for c in calls), "unconfirmed_requests": sum(not c.get("physical_call_confirmed", True) for c in calls), "stages": stages, "solver_executions": len(executions), "solver_seconds_observed_sum": sum(durations) if durations else (0 if not executions else None), "solver_seconds_missing": len(executions) - len(durations)}


def paired_cost_differences(pairs):
    """Compare recorded policy totals, with shared work in each counterfactual policy."""
    result = []
    for pair in pairs:
        if pair["state"] != "complete":
            continue
        costs = {mode: pair["costs"][mode + "_including_shared"] for mode in MODES}
        stages = {mode: {(r["stage"], r["backend"]): r for r in costs[mode]["stages"]} for mode in MODES}
        keys = set(stages["plain"]) | set(stages["reflection"])
        base = {k: pair[k] for k in ("model", "suite", "problem", "instance", "size")}
        for stage, backend in sorted(keys):
            rows = {mode: stages[mode].get((stage, backend), {}) for mode in MODES}
            fields = set(rows["plain"].get("raw_usage_fields", {})) | set(rows["reflection"].get("raw_usage_fields", {}))
            for field in sorted(fields | {"elapsed_seconds"}):
                values = {}
                for mode, row in rows.items():
                    if not row:
                        values[mode] = 0
                    elif field == "elapsed_seconds":
                        values[mode] = row["elapsed_seconds_observed_sum"] if row["elapsed_seconds_missing_calls"] == 0 else None
                    else:
                        usage = row["raw_usage_fields"].get(field, {})
                        values[mode] = usage.get("observed_sum") if usage.get("complete") else None
                result.append(base | {"stage": stage, "backend": backend, "measure": field, "plain": values["plain"], "reflection": values["reflection"], "reflection_minus_plain": values["reflection"] - values["plain"] if all(v is not None for v in values.values()) else None})
        solver_values = {mode: c["solver_seconds_observed_sum"] if c["solver_seconds_missing"] == 0 else None for mode, c in costs.items()}
        result.append(base | {"stage": "solver", "backend": "local", "measure": "execution_seconds", "plain": solver_values["plain"], "reflection": solver_values["reflection"], "reflection_minus_plain": solver_values["reflection"] - solver_values["plain"] if all(v is not None for v in solver_values.values()) else None})
    return result


def paired_statistics(pairs, budget):
    """Resample whole case pairs, retaining paired outcomes across all budgets."""
    groups = defaultdict(list)
    for pair in pairs:
        if pair["state"] != "complete":
            continue
        for dimension, group in (("overall", "all"), ("suite", pair["suite"]), ("problem", pair["suite"] + "/" + pair["problem"]), ("size", pair["size"])):
            groups[(pair["model"], dimension, group)].append(pair)
    result = []
    for (model, dimension, group), rows in sorted(groups.items()):
        n = len(rows)
        seed = BOOTSTRAP_SEED + int(hashlib.sha256(f"{model}/{dimension}/{group}".encode()).hexdigest()[:8], 16)
        indices = np.random.default_rng(seed).integers(0, n, size=(BOOTSTRAP_RESAMPLES, n)) if n > 1 else None
        for k in range(1, budget + 1):
            for metric in OUTCOMES:
                plain = np.array([r["policies"]["plain"]["attempt_metrics"][str(k)][metric] for r in rows], dtype=int)
                reflected = np.array([r["policies"]["reflection"]["attempt_metrics"][str(k)][metric] for r in rows], dtype=int)
                delta = reflected - plain
                interval = np.quantile(delta[indices].mean(axis=1), [0.025, 0.975]).tolist() if indices is not None else None
                result.append({"model": model, "dimension": dimension, "group": group, "budget": k, "metric": metric, "paired_cases": n, "plain_successes": int(plain.sum()), "reflection_successes": int(reflected.sum()), "plain_rate": float(plain.mean()), "reflection_rate": float(reflected.mean()), "reflection_minus_plain": float(delta.mean()), "plain_only": int(((plain == 1) & (reflected == 0)).sum()), "reflection_only": int(((plain == 0) & (reflected == 1)).sum()), "both": int(((plain == 1) & (reflected == 1)).sum()), "neither": int(((plain == 0) & (reflected == 0)).sum()), "paired_bootstrap_95": interval, "interval_degenerate": bool(interval and interval[0] == interval[1])})
    return result


def build_campaign_report(root: Path, models=None):
    """Audit one campaign root read-only; `models` limits the report to a selection such as PAPER_MODELS."""
    snapshot = Snapshot(root)
    root = snapshot.root
    manifest = snapshot.json(root / "campaign.json", True)
    if not manifest:
        raise ValueError("Readable campaign.json required")
    errors = []
    if identity({k: v for k, v in manifest.items() if k != "protocol_id"}) != manifest.get("protocol_id"):
        errors.append({"path": "campaign.json", "issue": "Frozen campaign identity differs"})
    controller_valid = bool(manifest.get("controller_sha256"))
    for name, digest in manifest.get("controller_sha256", {}).items():
        if not snapshot.hash_matches(root / "controller" / name, digest):
            errors.append({"path": "controller/" + name, "issue": "Frozen controller changed"})
            controller_valid = False
    inventory = manifest.get("instances", [])
    identities = [(c["suite"], c["problem"], c["instance"]) for c in inventory]
    if len(identities) != len(set(identities)) or any(c.get("size") not in SIZES for c in inventory):
        raise ValueError("Frozen inventory has duplicate case identities or unsupported sizes")
    for case in inventory:
        for name, digest in case["sha256"].items():
            path = root / "dataset" / case["suite"] / case["problem"] / case["instance"] / name
            if not snapshot.hash_matches(path, digest):
                errors.append({"path": snapshot.relative(path), "issue": "Frozen dataset changed"})
    models = default_models(manifest) if models is None else models
    skipped = [s for s in queue_for(manifest) if not reported(s["model"], models)]
    specifications = [s for s in queue_for(manifest) if reported(s["model"], models)]
    phase_pointer = snapshot.json(root / "active_execution_phase.json")
    if phase_pointer and phase_pointer.get("phase_file"):
        phase_path = Path(phase_pointer["phase_file"])
        snapshot.json(phase_path if phase_path.is_absolute() else root / phase_path, True)
    try:
        approvals = approved_receipts(root)
    except TerminalCompletionError as error:
        errors.append({"path": "active_execution_phase.json", "issue": str(error)})
        approvals = {}
    pairs = []
    checks = []
    for specification in specifications:
        for case in inventory:
            pair = {"model": specification["model"], "backend": specification["backend"], **{k: case[k] for k in ("suite", "problem", "instance", "size", "seed")}, "policies": {mode: audit_policy(snapshot, manifest, specification, case, mode, approvals, checks) for mode in MODES}}
            first = [snapshot.json(root / p["path"] / "shared_first.json") for p in pair["policies"].values()]
            if all(first):
                shared_id = identity({"protocol": manifest["protocol_id"], "model": pair["model"], "instance": case["instance"], "attempt": 1})
                # The recorded path is absolute; compare its part inside the campaign so a moved copy still audits.
                expected_canonical = Path(pair["policies"]["plain"]["path"]) / "attempts/attempt_01"
                recorded = Path(first[0].get("canonical", "")).parts
                if first[0] != first[1] or first[0].get("id") != shared_id or recorded[-len(expected_canonical.parts):] != expected_canonical.parts:
                    for policy in pair["policies"].values():
                        policy["audit_issues"].append("Shared-first identity differs between policies")
                first_scores = [score_base(p["attempts"][0]) if p["attempts"] else None for p in pair["policies"].values()]
                if all(first_scores) and first_scores[0] != first_scores[1]:
                    for policy in pair["policies"].values():
                        policy["audit_issues"].append("Shared-first scored outcomes differ between policies")
                if first_scores[0]:
                    for stage, key in (("generation", "generation_invocation_id"), ("algorithm_verification", "judge_invocation_id")):
                        if first_scores[0].get("metrics", {}).get(stage, {}).get("invocation_id") != first[0].get(key):
                            for policy in pair["policies"].values():
                                policy["audit_issues"].append("Shared-first physical invocation differs")
            elif any(p["attempts"] for p in pair["policies"].values()):
                for policy in pair["policies"].values():
                    policy["audit_issues"].append("Shared-first evidence incomplete")
            pairs.append(pair)
    if checks:
        if controller_valid:
            try:
                checked = frozen_output_checks(root, [{k: v for k, v in c.items() if k not in ("row", "attempt", "objective")} for c in checks])
                for check, result in zip(checks, checked, strict=True):
                    if not result.get("ok") or not finite(result.get("objective_value")) or not math.isclose(result["objective_value"], check["objective"], rel_tol=1e-6, abs_tol=1e-6):
                        check["row"]["audit_issues"].append(f"Attempt {check['attempt']}: independent output verification differs: {result.get('issues')}")
            except (OSError, ValueError, subprocess.SubprocessError) as error:
                errors.append({"path": "controller", "issue": "Frozen verifier failed: " + str(error)})
        else:
            errors.append({"path": "controller", "issue": "Output recheck requires intact frozen controller hashes"})
    calls, executions, cost_errors = scan_costs(snapshot, manifest, pairs)
    errors.extend(cost_errors)
    status = snapshot.json(root / "status.json") or {}
    execution_scope = {key: status.get(key) for key in ("state", "active_model", "execution_scope", "scope_expected_pairs", "scope_complete", "campaign_complete", "started_at")}
    window = snapshot.finish()
    for pair in pairs:
        for policy in pair["policies"].values():
            for error in window["read_errors"] + errors:
                if error["path"].startswith(policy["path"] + "/"):
                    policy["audit_issues"].append(error["path"] + ": " + error["issue"])
            if policy["audit_issues"]:
                policy["state"] = "audit_failed"
            if any(path.startswith(policy["path"] + "/") for path in window["changed_during_read"]):
                policy["state"] = "changed_during_snapshot"
            policy.pop("attempts")
        states = [p["state"] for p in pair["policies"].values()]
        pair["state"] = "complete" if all(s == "complete" for s in states) else "audit_failed" if "audit_failed" in states else "pending" if all(s == "pending" for s in states) else "partial"
        if any(not e["path"].startswith("runs/") for e in errors + window["read_errors"]):
            for policy in pair["policies"].values():
                if policy.get("actual_attempts"):
                    policy["state"] = "audit_failed"
                    policy["audit_issues"].append("Campaign-wide immutable evidence audit failed")
            pair["state"] = "audit_failed" if any(p.get("actual_attempts") for p in pair["policies"].values()) else pair["state"]
        if pair["state"] == "audit_failed":
            for policy in pair["policies"].values():
                if policy["state"] == "complete":
                    policy["state"] = "paired_audit_failed"
        selected = [c for c in calls if c["model"] == pair["model"] and c["instance"] == pair["instance"]]
        solvers = [e for e in executions if e["model"] == pair["model"] and e["instance"] == pair["instance"]]
        shared_calls = [c for c in selected if c["classification"] == "shared_first"]
        shared_solver = [e for e in solvers if e["mode"] == "shared"]
        pair["costs"] = {"shared_first": cost_summary(shared_calls, shared_solver), "physical_total": cost_summary(selected, solvers)}
        for mode in MODES:
            own = [c for c in selected if c["classification"] == "policy" and c["mode"] == mode]
            own_solver = [e for e in solvers if e["mode"] == mode]
            pair["costs"][mode + "_incremental"] = cost_summary(own, own_solver)
            pair["costs"][mode + "_including_shared"] = cost_summary(shared_calls + own, shared_solver + own_solver)
        if selected and pair["state"] == "pending":
            pair["state"] = "partial"
        pair["costs"]["additional_calls"] = cost_summary([c for c in selected if c["classification"] not in ("shared_first", "policy")])
    completed = sum(p["state"] == "complete" for p in pairs)
    complete = bool(pairs) and completed == len(pairs) and not errors and not window["read_errors"] and not window["changed_during_read"]
    size_rows = [{"size": size, "inventory_cases": sum(c["size"] == size for c in inventory), "expected_pairs": sum(p["size"] == size for p in pairs), "completed_pairs": sum(p["size"] == size and p["state"] == "complete" for p in pairs)} for size in SIZES]
    assert sum(r["expected_pairs"] for r in size_rows) == len(pairs)
    assert sum(r["completed_pairs"] for r in size_rows) == completed
    by_class = {key: cost_summary([c for c in calls if c["classification"] == key]) for key in sorted({c["classification"] for c in calls})}
    eligibility = run_identity(manifest)
    report = {"schema_version": 2, "layout": "native_campaign", "generated_at_utc": now(), "input_root": str(root), "snapshot": window, "provisional": not complete, "cohort": eligibility, "cohort_manifest": manifest, "configuration": {k: manifest.get(k) for k in ("protocol_id", "max_attempts", "workers", "solver_timeout", "solver_threads", "agent_timeout", "judge_model")} | {"gap_threshold": manifest.get("gap_threshold", 0.01), "threshold_source": "manifest" if "gap_threshold" in manifest else "legacy native runner default, checked against each policy summary"}, "coverage": {"expected_data_instances": len(inventory), "expected_pairs": len(pairs), "completed_pairs": completed, "expected_instance_runs": len(pairs) * 2, "completed_instance_runs": sum(p["state"] == "complete" for pair in pairs for p in pair["policies"].values()), "pair_states": dict(Counter(p["state"] for p in pairs)), "complete": complete, "sizes": size_rows}, "skipped_models": [s["model"] for s in skipped], "audit_errors": errors, "paired_curves": paired_statistics(pairs, manifest["max_attempts"]), "pairs": pairs, "paired_cost_differences": paired_cost_differences(pairs), "calls": calls, "solver_executions": executions, "costs": {"all_recorded_physical": cost_summary(calls, executions), "by_classification": by_class}, "execution_scope": execution_scope, "elapsed_time": {"campaign_created_at": manifest.get("created_at"), "snapshot_end_utc": window["finished_at_utc"], "active_execution_seconds": None, "note": "Stage durations are cumulative work. Active elapsed time is not reconstructed: parallel execution, waiting, cooldown, restart and user pause intervals require a complete event timeline."}, "methods": {"primary_endpoint": "Cumulative verified objective-quality success by attempt, on the same completed model/case pairs for both policies and every budget.", "interval": "95% paired case percentile bootstrap; whole pairs resampled together, 4000 draws, deterministic seed 20261013 plus group hash. Describes case variation conditional on one saved run per case; no interval for fewer than two pairs. Degenerate intervals are flagged.", "interval_source": "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html", "cost_attribution": "Each policy includes its shared first call/execution for paired policy cost; physical totals count that shared event once. Additional failed/rejected and uncommitted calls are separate. Raw token fields stay separated by backend; cache, reasoning and total fields are never added to input/output. Request-only records before the call gate are start-unconfirmed and excluded from proven physical counts.", "missing_usage": "Missing usage is null. Observed sums include only recorded values and retain the number of missing calls for every field.", "live_denominators": "Unfinished and changing pairs have no final outcome; primary paired curves use only fully completed audited pairs. Provisional cohorts reflect execution order and can be selected toward early successes."}}
    return report


def _write_csv(path, rows, fields):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value) if isinstance(value, (dict, list)) else value for key, value in row.items() if key in fields})


def export_campaign_report(report, destination):
    destination = Path(destination).resolve()
    root = Path(report["input_root"]).resolve()
    if destination.is_relative_to(root):
        raise ValueError("Derived reports must be outside the raw campaign directory")
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "evaluation.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    curve_fields = ["model", "dimension", "group", "budget", "metric", "paired_cases", "plain_successes", "reflection_successes", "plain_rate", "reflection_rate", "reflection_minus_plain", "plain_only", "reflection_only", "both", "neither", "paired_bootstrap_95", "interval_degenerate"]
    _write_csv(destination / "paired_curves.csv", report["paired_curves"], curve_fields)
    _write_csv(destination / "physical_calls.csv", report["calls"], ["invocation_id", "model", "suite", "problem", "instance", "size", "mode", "attempt", "stage", "backend", "requested_model", "classification", "ok", "terminal_state", "reflection_delivery_failure", "usage_available", "usage", "elapsed_seconds", "path", "evidence", "physical_call_confirmed"])
    _write_csv(destination / "case_costs.csv", report["pairs"], ["model", "suite", "problem", "instance", "size", "state", "costs"])
    _write_csv(destination / "paired_cost_differences.csv", report["paired_cost_differences"], ["model", "suite", "problem", "instance", "size", "stage", "backend", "measure", "plain", "reflection", "reflection_minus_plain"])
    _write_csv(destination / "solver_executions.csv", report["solver_executions"], ["solver_id", "model", "instance", "mode", "attempt", "execution_seconds", "path"])
    _write_csv(destination / "coverage_by_size.csv", report["coverage"]["sizes"], ["size", "inventory_cases", "expected_pairs", "completed_pairs"])
    state = "PROVISIONAL" if report["provisional"] else "COMPLETE"
    coverage = report["coverage"]
    lines = [f"# DecompGym paired report — {state}", "", f"Snapshot: {report['snapshot']['started_at_utc']} to {report['snapshot']['finished_at_utc']}. Cohort protocol: `{report['configuration']['protocol_id']}`. Source: `{report['input_root']}`.", "", f"Audited completed pairs: **{coverage['completed_pairs']}/{coverage['expected_pairs']}**; completed policy runs: {coverage['completed_instance_runs']}/{coverage['expected_instance_runs']}. Pair states: `{json.dumps(coverage['pair_states'], sort_keys=True)}`. Frozen inventory: {coverage['expected_data_instances']} cases.", "", report["methods"]["live_denominators"], "", "| Model | Completed / planned pairs | Q@1 Plain / Reflection | Q@5 Plain / Reflection | Q@10 Plain / Reflection |", "| --- | ---: | --- | --- | --- |"]
    for model in dict.fromkeys(p["model"] for p in report["pairs"]):
        rows = [p for p in report["pairs"] if p["model"] == model]
        values = []
        for budget in (1, 5, 10):
            point = next((c for c in report["paired_curves"] if c["model"] == model and c["dimension"] == "overall" and c["budget"] == budget and c["metric"] == "quality"), None)
            values.append(f"{point['plain_successes']}/{point['paired_cases']} / {point['reflection_successes']}/{point['paired_cases']}" if point else "Pending")
        lines.append(f"| {model} | {sum(p['state'] == 'complete' for p in rows)} / {len(rows)} | " + " | ".join(values) + " |")
    lines += ["", "| Size | Frozen cases | Planned pairs | Completed pairs |", "| --- | ---: | ---: | ---: |"]
    lines += [f"| {r['size']} | {r['inventory_cases']} | {r['expected_pairs']} | {r['completed_pairs']} |" for r in coverage["sizes"]]
    lines += ["", "Size counts reconcile to the complete frozen inventory and paired denominators. Family and size curves, discordant pairs and paired intervals are in `paired_curves.csv`.", "", "| Stage / backend | Physical calls | Usage present / missing | Observed client seconds | Missing duration |", "| --- | ---: | ---: | ---: | ---: |"]
    for stage in report["costs"]["all_recorded_physical"]["stages"]:
        duration = stage["elapsed_seconds_observed_sum"]
        lines.append(f"| {stage['stage']} / {stage['backend']} | {stage['calls']} | {stage['usage_available_calls']} / {stage['usage_missing_calls']} | {duration:.2f} | {stage['elapsed_seconds_missing_calls']} |" if duration is not None else f"| {stage['stage']} / {stage['backend']} | {stage['calls']} | {stage['usage_available_calls']} / {stage['usage_missing_calls']} | Missing | {stage['elapsed_seconds_missing_calls']} |")
    lines += ["", "| Stage / backend | Raw token field | Observed tokens | Calls with field / missing |", "| --- | --- | ---: | ---: |"]
    for stage in report["costs"]["all_recorded_physical"]["stages"]:
        for key, field in stage["raw_usage_fields"].items():
            lines.append(f"| {stage['stage']} / {stage['backend']} | {key} | {field['observed_sum']} | {field['available_calls']} / {field['missing_calls']} |")
    costs = report["costs"]["all_recorded_physical"]
    lines += ["", f"Proven physical invocation records: {costs['physical_cli_calls']}; additional request records with start unconfirmed: {costs['unconfirmed_requests']}."]
    lines += ["", f"Unique solver executions: {costs['solver_executions']}; recorded solver seconds: {costs['solver_seconds_observed_sum']}; missing durations: {costs['solver_seconds_missing']}.", "", report["methods"]["cost_attribution"], "", report["methods"]["missing_usage"], "", report["elapsed_time"]["note"], "", report["methods"]["primary_endpoint"], "", report["methods"]["interval"] + " [Paired bootstrap method](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html).", "", "Each pair compares the two complete retry policies: plain regeneration versus retries using prior code, feedback and reflection. Dataset seeds vary cases; each case has one saved model run. Model execution follows the planned order, so model, time and configuration remain linked.", "", f"Audit issues: {len(report['audit_errors'])}; read issues: {len(report['snapshot']['read_errors'])}; files changed during read: {len(report['snapshot']['changed_during_read'])}. Policy-specific issues are in `evaluation.json`.", "", "`evaluation.json` contains frozen cohort identity, evidence paths, per-policy outcomes, physical costs and missing-value counts. `physical_calls.csv` retains raw backend usage; `case_costs.csv` separates shared, incremental and additional calls; `paired_cost_differences.csv` gives within-case stage/token differences only where both totals are known; `solver_executions.csv` deduplicates the shared execution."]
    if report.get("execution_scope", {}).get("execution_scope"):
        lines += ["", "Current authorized execution scope: `" + json.dumps(report["execution_scope"]["execution_scope"]) + "`; scope pairs: " + str(report["execution_scope"].get("scope_expected_pairs")) + ". The scientific cohort denominator remains the complete frozen model queue."]
    if report["skipped_models"]:
        lines += ["", "Models of this campaign outside the selection: " + ", ".join(report["skipped_models"]) + ". Their raw data remain in the source campaign."]
    (destination / "report.md").write_text("\n".join(lines) + "\n")
