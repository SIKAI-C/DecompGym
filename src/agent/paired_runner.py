"""Resumable native paired retries with a single shared first evaluation."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re
import shutil
import time
import uuid
from datetime import datetime, timezone

from src.agent.codex_client import CodexClient
from src.agent.algorithm_verifier import algorithm_review_prompt
from src.agent.exec_logger import ExecLogger
from src.agent.experiment_runner import AttemptResult, ExperimentRunner
from src.agent.files_manager import ArtifactManager
from src.agent.output_verifier import OutputVerifier
from src.agent.prompt_builder import PromptBuilder
from src.agent.prompt_policy import FEEDBACK_POLICY, WORKSPACE_POLICY
from src.agent.runtime_state import runtime_lock


class InfrastructureError(RuntimeError):
    """Retry the pending stage without advancing the scored attempt number."""


class ReflectionArtifactError(InfrastructureError):
    """AGY completed a turn but omitted the required reflection file."""


class ReflectionDeliveryFailure(RuntimeError):
    """A clarified call wrote the reflection to a proven wrong target."""


class ProtocolError(RuntimeError):
    """An immutable protocol or artifact changed; do not retry automatically."""


def save(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def read(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def digest(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def solver_infrastructure_error(attempt_dir, stderr):
    """Separate a generated solver's bad path from a failed Python launch."""
    text = stderr.lower()
    if any(marker in text for marker in ("license", "failed to connect", "network is unreachable")):
        return True
    if "no such file or directory: '" not in text:
        return False
    lines = stderr.strip().splitlines()
    frames = re.findall(r'^  File "([^"]+)", line \d+', stderr, re.MULTILINE)
    solver = attempt_dir / "solver.py"
    if not lines or not lines[-1].startswith("FileNotFoundError: [Errno 2] No such file or directory:") or not frames or Path(frames[-1]).resolve() != solver.resolve() or "Traceback (most recent call last):" not in stderr:
        return True
    instance = attempt_dir.parent.parent
    case = read(instance / "identity.json") or {}
    generation = read(attempt_dir / "agent_receipt.json") or {}
    for name in ("data.json", "prompt.txt"):
        path = instance / name
        if not path.is_file() or digest(path) != case.get("inputs", {}).get(name):
            return True
    return not solver.is_file() or digest(solver) != generation.get("result", {}).get("artifact_sha256")


class PairedRunner(ExperimentRunner):
    reflection_retry_delays = (5, 10)

    def __init__(self, *, client, judge_client, reflection_client=None, protocol_id="native-qualitative-v2", on_stage=None, call_lock=None, **kwargs):
        super().__init__(client=client, **kwargs)
        self.judge_client = judge_client
        self.reflection_client = reflection_client or client
        self.protocol_id = protocol_id
        self.on_stage = on_stage or (lambda **kwargs: None)
        self.call_lock = call_lock
        self.mode = "plain"

    def edit(self, client, target, prompt, context, step):
        """Commit each physical call once; retain all failed-call evidence."""
        directory = target.parent
        request = {"protocol": self.protocol_id, "prompt_policies": {"feedback": FEEDBACK_POLICY, "workspace": WORKSPACE_POLICY}, "step": step, "prompt": prompt, "target": target.name, "context": [{"label": f"{p.parent.name}/{p.name}", "sha256": digest(p)} for p in context if p.exists()]}
        fingerprint = identity(request)
        receipt_path = directory / f"{step}_receipt.json"
        cached = read(receipt_path)
        if cached:
            if cached["request_sha256"] != fingerprint:
                raise ProtocolError("Saved CLI request differs from the current request")
            if cached["result"]["ok"] and (not target.exists() or digest(target) != cached["result"]["artifact_sha256"]):
                raise ProtocolError("A committed CLI artifact changed")
            if cached["result"].get("file_delivery_failure"):
                self.commit_delivery_failure(directory / "calls" / cached["result"]["invocation_id"], target, step, fingerprint, cached["result"])
            return cached["result"]
        calls = directory / "calls"
        calls.mkdir(exist_ok=True)
        # Recover the client commit even if the controller crashed before its receipt.
        for call in sorted(calls.iterdir()):
            prior = read(call / "request.json")
            metrics = read(call / f"{step}_metrics.json")
            if not prior or prior.get("request_sha256") != fingerprint or not metrics or (call / "rejected.json").exists():
                continue
            self.commit_delivery_failure(call, target, step, fingerprint, metrics)
            if metrics.get("ok"):
                if not target.exists() or digest(target) != metrics.get("artifact_sha256"):
                    raise ProtocolError("Completed CLI call lost its committed artifact")
                metrics["invocation_id"] = call.name
                save(receipt_path, {"request_sha256": fingerprint, "result": metrics})
                self.publish_call(call, directory, step)
                return metrics
        call = calls / uuid.uuid4().hex
        call.mkdir()
        save(call / "request.json", request | {"request_sha256": fingerprint, "invocation_id": call.name})
        target.write_text("")
        self.on_stage(mode=self.mode, attempt=int(directory.name.rsplit("_", 1)[1]), phase=step)
        lock = {"agy-cli": "native-agy-auth", "claude-code-cli": "native-claude-auth"}.get(client.protocol().get("backend"), "codex-native-auth")
        with self.call_lock(client) if self.call_lock else runtime_lock(lock):
            result = client.edit_file(target, prompt, context=context, log_dir=call, step=step)
        metrics = asdict(result)
        if result.ok and not metrics.get("artifact_sha256"):
            metrics["artifact_sha256"] = digest(target)
        metrics["invocation_id"] = call.name
        save(call / f"{step}_metrics.json", metrics)
        self.publish_call(call, directory, step)
        self.commit_delivery_failure(call, target, step, fingerprint, metrics)
        if not result.ok and not (step == "agent" and result.terminal_state == "completed" and result.returncode == 0):
            if step == "reflection" and client.protocol().get("backend") == "agy-cli" and result.terminal_state == "completed" and result.returncode == 0 and (result.error or "").startswith("AGY did not complete an edit (exit 0)"):
                diagnostics = "\n".join([result.error or "", *result.warnings, *(p.read_text(errors="replace") for p in (call / f"{step}_stderr.log", call / f"{step}_events.jsonl") if p.exists())]).lower()
                access_markers = ("unauthorized", "authentication", "not logged in", "access token", "refresh token", "refresh_token", "rate limit", "rate_limit", "quota", "resource_exhausted", "http 429", "status code 429")
                if not any(marker in diagnostics for marker in access_markers) and (not target.exists() or not target.read_text().strip()):
                    raise ReflectionArtifactError(f"{self.model} {step}: {result.error}")
            raise InfrastructureError(f"{self.model} {step}: {result.error}")
        save(receipt_path, {"request_sha256": fingerprint, "result": metrics})
        return metrics

    def commit_delivery_failure(self, call, target, step, fingerprint, metrics):
        if step != "reflection" or not (call / "reflection_delivery_failure.json").exists():
            return
        from src.agent.policy_completion import TerminalCompletionError, campaign_root_for, validate_delivery_call
        try:
            validate_delivery_call(call, campaign_root_for(target.parent))
        except TerminalCompletionError as error:
            raise ProtocolError(str(error)) from error
        metrics = dict(metrics, invocation_id=call.name, file_delivery_failure=True)
        save(call / "reflection_metrics.json", metrics)
        save(target.parent / "reflection_receipt.json", {"request_sha256": fingerprint, "result": metrics})
        self.publish_call(call, target.parent, step)
        raise ReflectionDeliveryFailure("The clarified reflection was written outside the required target")

    @staticmethod
    def publish_call(call, directory, step):
        for source in call.glob(f"{step}_*"):
            if source.is_file():
                shutil.copyfile(source, directory / source.name)

    def generation_prompt(self, instance):
        prompt = PromptBuilder.ensure_location_prompt((instance / "prompt.txt").read_text())
        prompt += "\n\nImplement a standalone solver.py using gurobipy and standard Python libraries. Write output.json using the documented contract. Do not import repository implementations or load reference solutions."
        prompt += f"\nThe harness limits the entire solver.py execution to {self.solver_timeout} seconds and numerical solvers to {self.solver_threads} thread(s). Design your algorithm and termination handling within this budget."
        return prompt

    def run_attempt(self, idx, attempt_dir, problem, gold):
        attempt_dir.mkdir(parents=True, exist_ok=True)
        saved = read(attempt_dir / "score_receipt.json")
        if saved:
            for name, expected in saved["artifact_sha256"].items():
                if not (attempt_dir / name).exists() or digest(attempt_dir / name) != expected:
                    raise ProtocolError(f"Scored artifact changed: {name}")
            return AttemptResult(**saved["result"])
        started = time.perf_counter()
        instance = attempt_dir.parent.parent
        context = [instance / "data.json"]
        if self.mode == "reflection" and idx > 1:
            context += [Path(p) for p in ArtifactManager.collect_previous_attempt_files(attempt_dir.parent, idx)]
        generation = self.edit(self.client, attempt_dir / "solver.py", self.generation_prompt(instance), context, "agent")
        metrics = {"generation": generation}
        if not generation["ok"]:
            result = AttemptResult(idx, "agent_failed", False, False, None, gold, None, [generation["error"] or "Model omitted the requested solver"], metrics=metrics)
        else:
            execution_path = attempt_dir / "execution.json"
            execution = read(execution_path)
            code_hash = digest(attempt_dir / "solver.py")
            if execution:
                if execution["solver_sha256"] != code_hash:
                    raise ProtocolError("Executed solver changed")
                if execution.get("output_sha256") and digest(attempt_dir / "output.json") != execution["output_sha256"]:
                    raise ProtocolError("Saved solver output changed")
            else:
                self.on_stage(mode=self.mode, attempt=idx, phase="solver")
                execution_started = time.perf_counter()
                exec_ok, stdout, stderr, output = self.run_solver(attempt_dir, self.solver_timeout, self.solver_threads)
                elapsed = time.perf_counter() - execution_started
                if not exec_ok and solver_infrastructure_error(attempt_dir, stderr):
                    save(attempt_dir / f"solver_infrastructure_{uuid.uuid4().hex}.json", {"stdout": stdout, "stderr": stderr, "solver_sha256": code_hash})
                    raise InfrastructureError("Native solver infrastructure unavailable; see saved execution diagnostics")
                execution = {"exec_ok": exec_ok, "stdout": stdout, "stderr": stderr, "output": output, "execution_seconds": elapsed, "solver_sha256": code_hash, "output_sha256": digest(attempt_dir / "output.json") if (attempt_dir / "output.json").exists() else None}
                save(execution_path, execution)
                save(attempt_dir / "solver_invocation.json", {"backend": "host-python", "timeout": self.solver_timeout, "threads": self.solver_threads, "solver_sha256": code_hash})
                if elapsed >= 5 and self.solver_cooldown_seconds:
                    self.on_stage(mode=self.mode, attempt=idx, phase="solver cooldown")
                    time.sleep(self.solver_cooldown_seconds)
            exec_ok, stdout, stderr, output = (execution[k] for k in ("exec_ok", "stdout", "stderr", "output"))
            metrics["execution_seconds"] = execution["execution_seconds"]
            metrics["execution_timed_out"] = "[framework] solver execution timed out" in stderr
            metrics["cooldown_seconds"] = self.solver_cooldown_seconds if execution["execution_seconds"] >= 5 else 0
            objective, issues, message = OutputVerifier.extract_objective(output)
            verification = {"ok": False, "issues": []}
            if exec_ok:
                if self.suite == "cg" and problem in {"csp", "gap", "shift"} and isinstance(output, dict) and "solution" in output and output["solution"] is None:
                    verification = {"ok": False, "issues": ["solution is null; no candidate solution supplied"]}
                else:
                    verification = OutputVerifier.verify_output(self.suite, problem, instance / "data.json", attempt_dir / "output.json", self.tol)
                issues.extend(verification.get("issues", []))
                if verification.get("objective_value") is not None:
                    objective = float(verification["objective_value"])
            else:
                issues.append(stderr or "solver execution failed")
            prompt = algorithm_review_prompt(self.suite)
            judged = self.edit(self.judge_client, attempt_dir / "alg_verifier.md", prompt, [attempt_dir / "solver.py"], "llm_verifier")
            notes = (attempt_dir / "alg_verifier.md").read_text().strip()
            verdict = {"VERDICT: PASS": "pass", "VERDICT: FAIL": "fail"}.get(notes.splitlines()[0].strip().upper() if notes else "")
            if verdict is None:
                save(attempt_dir / "calls" / judged["invocation_id"] / "rejected.json", {"reason": "malformed algorithm verdict"})
                (attempt_dir / "llm_verifier_receipt.json").unlink()
                raise InfrastructureError("Algorithm judge produced a malformed verdict")
            metrics["algorithm_verification"] = judged
            if verdict != "pass":
                issues.append("algorithm_verifier_failed: " + verdict)
            verify_ok = bool(verification["ok"]) and verdict == "pass"
            status, gap = OutputVerifier.assess_status(exec_ok=exec_ok, output=output, verify_ok=verify_ok, objective=objective, gold_obj=gold, gap_threshold=self.gap_threshold, execution_timed_out=metrics["execution_timed_out"])
            result = AttemptResult(idx, status, exec_ok, verify_ok, objective, gold, gap, issues, verdict, notes, stdout=stdout, stderr=stderr, feasibility_ok=bool(verification["ok"]), metrics=metrics)
            lines = ExecLogger.build_exec_lines(stdout, stderr)
            ExecLogger.append_output_message(lines, message)
            ExecLogger.append_verification_issues(lines, issues)
            ExecLogger.append_llm_verifier(lines, notes)
            ExecLogger.write_exec_log(attempt_dir, lines)
        metrics["total_seconds"] = time.perf_counter() - started
        artifacts = {name: digest(attempt_dir / name) for name in ("solver.py", "output.json", "alg_verifier.md", "exec.log") if (attempt_dir / name).exists()}
        save(attempt_dir / "score_receipt.json", {"result": result.to_dict(), "artifact_sha256": artifacts})
        save(attempt_dir / "summary.json", result.to_dict())
        return result

    def ensure_generation_failure_log(self, attempt_dir, result):
        """Supply recorded harness failure facts to an uncommitted reflection."""
        if result.status != "agent_failed" or (attempt_dir / "reflection_receipt.json").exists():
            return
        # Let edit() recover and validate a successful physical call unchanged.
        for metrics_path in (attempt_dir / "calls").glob("*/reflection_metrics.json"):
            metrics = read(metrics_path)
            if metrics and metrics.get("ok") and not (metrics_path.parent / "rejected.json").exists():
                return
        log_path = attempt_dir / "exec.log"
        provenance_path = attempt_dir / "generation_failure_context.json"
        if log_path.exists() and not provenance_path.exists():
            return
        score_path, generation_path = attempt_dir / "score_receipt.json", attempt_dir / "agent_receipt.json"
        score, receipt = read(score_path), read(generation_path)
        if not score or not receipt or score["result"]["status"] != "agent_failed" or receipt["result"].get("ok") is not False:
            raise ProtocolError("Generation failure feedback requires committed failure receipts")
        generation = score["result"]["metrics"]["generation"]
        if generation != receipt["result"]:
            raise ProtocolError("Generation failure feedback receipts differ")
        for name, expected in score["artifact_sha256"].items():
            path = attempt_dir / name
            if not path.is_file() or digest(path) != expected:
                raise ProtocolError("Scored artifact changed before failure feedback: " + name)
        if any((attempt_dir / name).exists() for name in ("execution.json", "solver_invocation.json", "output.json", "alg_verifier.md", "llm_verifier_receipt.json")):
            raise ProtocolError("Generation failure feedback conflicts with execution or judge artifacts")
        solver = attempt_dir / "solver.py"
        if solver.exists() and solver.read_text().strip():
            return
        contents = "\n".join([
            "[framework generation diagnostic]",
            "The generation call did not deliver the required solver.py in the task workspace.",
            "The required solver.py is empty or absent.",
            "Recorded generation diagnostic: " + str(generation.get("error") or "No usable solver artifact was delivered."),
            "Solver execution was not performed; there is no solver stdout or stderr.",
            "Algorithm judgment was not performed; alg_verifier.md is unavailable.",
            "This is framework feedback about the generation stage, not solver execution output or a model reflection.",
        ]) + "\n"
        provenance = {"kind": "framework_generation_failure_feedback", "score_receipt_sha256": digest(score_path), "generation_receipt_sha256": digest(generation_path), "solver_sha256": digest(solver) if solver.exists() else None, "exec_log_sha256": hashlib.sha256(contents.encode()).hexdigest(), "reflection_context_changed": True, "candidate_score_changed": False}
        existing = read(provenance_path)
        if provenance_path.exists() and existing != provenance:
            raise ProtocolError("Generation failure feedback provenance changed")
        if log_path.exists() and digest(log_path) != provenance["exec_log_sha256"]:
            raise ProtocolError("Generation failure feedback log changed")
        if not existing:
            save(provenance_path, provenance)
        if not log_path.exists():
            log_path.write_text(contents)

    def ensure_reflection(self, attempt_dir, result):
        self.ensure_generation_failure_log(attempt_dir, result)
        instance = attempt_dir.parent.parent
        prompt = PromptBuilder.build_reflection_prompt(self.generation_prompt(instance), objective_quality_failed=self.objective_quality_failed(result))
        prompt += "\nKeep the reflection concise and actionable."
        context = [instance / "data.json"] + [attempt_dir / name for name in ("solver.py", "exec.log", "alg_verifier.md", "output.json")]
        for retry in range(len(self.reflection_retry_delays) + 1):
            try:
                reflected = self.edit(self.reflection_client, attempt_dir / "debug.md", prompt, context, "reflection")
                break
            except ReflectionArtifactError:
                if retry == len(self.reflection_retry_delays):
                    raise
                delay = self.reflection_retry_delays[retry]
                failed = read(attempt_dir / "reflection_metrics.json")
                request = read(attempt_dir / "calls" / failed["invocation_id"] / "request.json")
                save(attempt_dir / f"reflection_retry_{uuid.uuid4().hex}.json", {"reason": "AGY completed without saving debug.md in the task workspace", "failed_invocation_id": failed["invocation_id"], "request_sha256": request["request_sha256"], "local_retry": retry + 1, "delay_seconds": delay, "recorded_at_unix": time.time()})
                self.on_stage(mode=self.mode, attempt=result.attempt, phase=f"reflection retry in {delay}s: missing debug.md")
                time.sleep(delay)
        result.metrics["reflection"] = reflected
        return result

    def prepare(self, source, root, mode):
        destination = root / mode / self.suite / self.problems[0] / source.name
        destination.mkdir(parents=True, exist_ok=True)
        expected = {"protocol_id": self.protocol_id, "prompt_policies": {"feedback": FEEDBACK_POLICY, "workspace": WORKSPACE_POLICY}, "model": self.model, "mode": mode, "max_attempts": self.max_attempts, "inputs": {name: digest(source / name) for name in ("data.json", "prompt.txt")}, "references": {name: digest(source / name) for name in ("meta.json", "gold_monolithic.json") if (source / name).exists()}}
        prior = read(destination / "identity.json")
        if prior and prior != expected:
            raise ProtocolError("Case protocol changed")
        for name, sha in expected["inputs"].items():
            path = destination / name
            if path.exists() and digest(path) != sha:
                raise ProtocolError("Case input changed")
            if not path.exists():
                shutil.copyfile(source / name, path)
        save(destination / "identity.json", expected)
        (destination / "attempts").mkdir(exist_ok=True)
        return destination

    def summary(self, source, destination, mode, results):
        value = {"instance": source.name, "suite": self.suite, "problem": self.problems[0], "source": str(source), "output_dir": str(destination), "backend": self.client.protocol().get("backend"), "model": self.model, "reflection": mode == "reflection", "solver_threads": self.solver_threads, "solver_timeout": self.solver_timeout, "solver_cooldown_seconds": self.solver_cooldown_seconds, "max_attempts": self.max_attempts, "gap_threshold": self.gap_threshold, "protocol_id": self.protocol_id, "shared_first_attempt": True, "complete": len(results) == self.max_attempts, "attempts": results}
        save(destination / "instance_summary.json", value)
        return value

    def run_pair(self, source, root):
        """The caller owns this (model, case) pair; policies never run concurrently."""
        from src.agent.policy_completion import DELIVERY_RECEIPT, TERMINAL_RECEIPT, TerminalCompletionError, approved_receipts, build_delivery_receipt, has_terminal_receipt, load_policy_summary, policy_is_complete
        saved = {mode: root / mode / self.suite / self.problems[0] / source.name for mode in ("plain", "reflection")}
        campaign_root = root.parent.parent if root.parent.name == "runs" and (root.parent.parent / "campaign.json").exists() else None
        try:
            approvals = approved_receipts(campaign_root) if campaign_root else {}
        except TerminalCompletionError as error:
            raise ProtocolError(str(error)) from error
        if any(has_terminal_receipt(directory) or (campaign_root and str((directory / TERMINAL_RECEIPT).relative_to(campaign_root)) in approvals) for directory in saved.values()):
            try:
                summaries = {mode: load_policy_summary(directory, campaign_root=campaign_root, approvals=approvals) for mode, directory in saved.items()}
                if all(policy_is_complete(summary, self.max_attempts, self.protocol_id) for summary in summaries.values()):
                    return summaries
                raise ProtocolError("An adjudicated pair is not fully closed")
            except TerminalCompletionError as error:
                raise ProtocolError(str(error)) from error
        destinations = {mode: self.prepare(source, root, mode) for mode in ("plain", "reflection")}
        gold = OutputVerifier.load_gold_objective(source, self.suite)
        self.mode = "plain"
        first_dir = destinations["plain"] / "attempts/attempt_01"
        first = self.run_attempt(1, first_dir, self.problems[0], gold)
        shared_id = identity({"protocol": self.protocol_id, "model": self.model, "instance": source.name, "attempt": 1})
        # Copies keep reflection notes out of the independent policy and make the old report layout usable.
        reflection_first = destinations["reflection"] / "attempts/attempt_01"
        reflection_first.mkdir(exist_ok=True)
        for path in first_dir.iterdir():
            if path.is_file() and path.name not in {"identity.json"}:
                target = reflection_first / path.name
                if not target.exists():
                    shutil.copyfile(path, target)
                elif path.name in {"solver.py", "output.json", "alg_verifier.md"} and digest(path) != digest(target):
                    raise ProtocolError("Shared first-attempt projection differs")
        for destination in destinations.values():
            save(destination / "shared_first.json", {"id": shared_id, "canonical": str(first_dir), "generation_invocation_id": first.metrics["generation"].get("invocation_id"), "judge_invocation_id": first.metrics.get("algorithm_verification", {}).get("invocation_id")})
            if not (destination / "instance_summary.json").exists():
                self.summary(source, destination, destination.parents[2].name, [first.to_dict()])
        summaries = {}
        for mode, destination in destinations.items():
            self.mode = mode
            results, successful, first_success = [], None, None
            for idx in range(1, self.max_attempts + 1):
                self.on_stage(mode=mode, attempt=idx, phase="checkpoint")
                directory = destination / "attempts" / f"attempt_{idx:02d}"
                if successful is not None:
                    result = AttemptResult(idx, "skipped_previous_solved", True, True, successful.objective, gold, successful.gap_to_gold, [], reused_from_previous=True, first_success_attempt=first_success, feasibility_ok=True)
                else:
                    result = deepcopy(first) if idx == 1 else self.run_attempt(idx, directory, self.problems[0], gold)
                    if self.can_early_stop(result, self.gap_threshold):
                        successful, first_success = result, idx
                    elif mode == "reflection":
                        # Scoring is already durable even when this separate stage fails.
                        try:
                            result = self.ensure_reflection(directory, result)
                        except ReflectionDeliveryFailure:
                            receipt = build_delivery_receipt(destination, directory, campaign_root, recorded_at=datetime.now(timezone.utc).isoformat())
                            save(destination / DELIVERY_RECEIPT, receipt)
                            summaries[mode] = load_policy_summary(destination, campaign_root=campaign_root, approvals=approvals)
                            return summaries
                results.append(result.to_dict())
                summaries[mode] = self.summary(source, destination, mode, results)
                print(f"{self.model} {source.name} {mode} attempt {idx}/{self.max_attempts}: {result.status}", flush=True)
        return summaries
