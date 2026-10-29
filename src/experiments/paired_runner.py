"""Resumable native paired retries with a single shared first evaluation."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
import hashlib
import math
from pathlib import Path
import shutil
import threading
import time
from typing import Any
import uuid

from src.agent.prompts.builder import PromptBuilder
from src.agent.prompts.judge import algorithm_review_prompt, parse_verdict
from src.agent.prompts.policy import FEEDBACK_POLICY, WORKSPACE_POLICY, REFLECTION_POLICY, reflection_delivery_feedback
from src.experiments.environment import require_runtime_environment
from src.experiments.exec_log import ExecLogger
from src.experiments.locks import runtime_lock
from src.experiments.sandbox import SolverLaunchError
from src.experiments.verification import OutputVerifier
from src.utils.errors import InfrastructureError, ProtocolError
from src.utils.records import digest, identity, read, save


SUITE_PROBLEMS = {"bd": ["tsso", "fctp", "km", "lrp", "ufl"], "cg": ["bp", "csp", "gap", "shift"]}
# Harder-family cohort (exploration): accepted when named explicitly, never part of the default list.
HARDER_SUITE_PROBLEMS = {"bd": ["lbbd"], "cg": ["gcol", "bppc", "vrptw", "pdptw", "nrp"]}


@dataclass
class AttemptResult:
    attempt: int
    status: str
    exec_ok: bool
    verify_ok: bool
    objective: float | None
    gold_objective: float | None
    gap_to_gold: float | None
    issues: list[str]
    llm_verdict: str | None = None
    llm_notes: str | None = None
    reused_from_previous: bool = False
    first_success_attempt: int | None = None
    stdout: str | None = None
    stderr: str | None = None
    feasibility_ok: bool = False
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def previous_attempt_files(attempts_root: Path, current_idx: int) -> list[Path]:
    """Read-only files of earlier attempts given to a reflection-policy generation."""
    files = []
    for idx in range(1, current_idx):
        directory = attempts_root / f"attempt_{idx:02d}"
        files.extend((directory / name).resolve() for name in ("solver.py", "debug.md", "reflection_feedback.md") if (directory / name).exists())
    return files


class PairedRunner:
    """Run one (model, instance) pair: a shared first attempt, then plain and reflection retries."""

    def __init__(self, *, client, judge_client, solver_executor, reflection_client=None, protocol_id="native-qualitative-v2", on_stage=None, call_lock=None,
                 model=None, suite="bd", problem=None, max_attempts=10, solver_timeout=60, agent_timeout=900, solver_threads=1,
                 solver_cooldown_seconds=0, tol=1e-6, gap_threshold=0.01):
        if suite not in SUITE_PROBLEMS:
            raise ValueError("Choose suite bd or cg.")
        if max_attempts < 1 or solver_threads < 1 or min(solver_timeout, agent_timeout, tol) <= 0 or gap_threshold < 0 or solver_cooldown_seconds < 0:
            raise ValueError("Attempt/time/tolerance budgets must be positive; gap threshold must be nonnegative.")
        self.runtime_environment = require_runtime_environment()
        self.model, self.suite = model, suite
        self.problems = list(problem) if problem else list(SUITE_PROBLEMS[suite])
        if any(p not in SUITE_PROBLEMS[suite] + HARDER_SUITE_PROBLEMS[suite] for p in self.problems):
            raise ValueError(f"Problems must belong to {suite}: {SUITE_PROBLEMS[suite]}")
        self.max_attempts, self.solver_timeout, self.solver_threads = max_attempts, solver_timeout, solver_threads
        self.solver_cooldown_seconds = solver_cooldown_seconds
        self.abort_event = threading.Event()
        self.tol, self.gap_threshold = tol, gap_threshold
        self.client = client
        self.judge_client = judge_client
        self.reflection_client = reflection_client or client
        self.protocol_id = protocol_id
        self.on_stage = on_stage or (lambda **kwargs: None)
        self.call_lock = call_lock
        self.run_solver = solver_executor
        self.mode = "plain"

    def objective_quality_failed(self, result: AttemptResult) -> bool:
        """Expose one qualitative check outcome, without passing the reference to the prompt."""
        values = (result.objective, result.gold_objective, result.gap_to_gold)
        return bool(result.exec_ok and result.feasibility_ok and not result.metrics.get("execution_timed_out")
                    and all(isinstance(value, (int, float)) and math.isfinite(value) for value in values)
                    and result.gap_to_gold > self.gap_threshold)

    @staticmethod
    def can_early_stop(result: AttemptResult, gap_threshold: float) -> bool:
        return result.status == "ok" and result.exec_ok and result.verify_ok and result.gap_to_gold is not None and result.gap_to_gold <= gap_threshold

    def edit(self, client, target, prompt, context, step):
        """Commit each physical call once; retain all failed-call evidence."""
        directory = target.parent
        request = {"protocol": self.protocol_id, "reflection_policy": REFLECTION_POLICY, "prompt_policies": {"feedback": FEEDBACK_POLICY, "workspace": WORKSPACE_POLICY}, "step": step, "prompt": prompt, "target": target.name, "context": [{"label": f"{p.parent.name}/{p.name}", "sha256": digest(p)} for p in context if p.exists()]}
        fingerprint = identity(request)
        receipt_path = directory / f"{step}_receipt.json"
        cached = read(receipt_path)
        if cached:
            if cached["request_sha256"] != fingerprint:
                raise ProtocolError("Saved CLI request differs from the current request")
            if cached["result"]["ok"] and (not target.exists() or digest(target) != cached["result"]["artifact_sha256"]):
                raise ProtocolError("A committed CLI artifact changed")
            if cached["result"].get("file_delivery_failure"):
                self.validate_reflection_omission(directory / "calls" / cached["result"]["invocation_id"], target, cached["result"])
            return cached["result"]
        calls = directory / "calls"
        calls.mkdir(exist_ok=True)
        # Recover the client commit even if the controller crashed before its receipt.
        for call in sorted(calls.iterdir()):
            prior = read(call / "request.json")
            metrics = read(call / f"{step}_metrics.json")
            if prior and prior.get("request_sha256") == fingerprint and not metrics and hasattr(client, "recover_file"):
                recovered = client.recover_file(target, log_dir=call, step=step)
                if recovered is not None:
                    metrics = asdict(recovered)
            if not prior or prior.get("request_sha256") != fingerprint or not metrics or (call / "rejected.json").exists():
                continue
            metrics = self.record_reflection_omission(call, target, step, metrics)
            completed_generation_failure = step == "agent" and metrics.get("terminal_state") == "completed" and metrics.get("returncode") == 0 and not metrics.get("ok")
            if metrics.get("ok") or metrics.get("file_delivery_failure") or completed_generation_failure:
                if not completed_generation_failure and (not target.exists() or digest(target) != metrics.get("artifact_sha256")):
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
        metrics = self.record_reflection_omission(call, target, step, metrics)
        save(call / f"{step}_metrics.json", metrics)
        self.publish_call(call, directory, step)
        if not result.ok and not metrics.get("file_delivery_failure") and not (step == "agent" and result.terminal_state == "completed" and result.returncode == 0):
            raise InfrastructureError(f"{self.model} {step}: {result.error}")
        save(receipt_path, {"request_sha256": fingerprint, "result": metrics})
        return metrics

    def record_reflection_omission(self, call, target, step, metrics):
        if step != "reflection":
            return metrics
        if metrics.get("file_delivery_failure"):
            self.validate_reflection_omission(call, target, metrics)
            return metrics
        if metrics.get("terminal_state") != "completed" or metrics.get("returncode") != 0 or metrics.get("ok"):
            return metrics
        diagnostics = "\n".join([metrics.get("error") or "", *metrics.get("warnings", []), *(p.read_text(errors="replace") for p in (call / "reflection_stderr.log",) if p.exists())]).lower()
        infrastructure_markers = ("unauthorized", "authentication", "not logged in", "access token", "refresh token", "refresh_token", "rate limit", "rate_limit", "quota", "resource_exhausted", "http 429", "status code 429", "permission denied", "operation not permitted", "failed to spawn", "failed to initialize", "connection refused", "network is unreachable", "timed out")
        if any(marker in diagnostics for marker in infrastructure_markers):
            return metrics
        if not target.exists() and not target.is_symlink():
            target.write_text("")
        if not target.is_file() or target.is_symlink() or target.read_bytes().strip():
            return metrics
        # A completed omission has one durable outcome, including misplaced-file evidence.
        evidence = [p for p in call.glob("reflection_*") if p.is_file() and p.suffix != ".jsonl" and not p.name.endswith("metrics.json")]
        evidence += [p for p in (call / "unexpected_outputs").rglob("*") if p.is_file() and not p.is_symlink()]
        metrics = dict(metrics, invocation_id=call.name, file_delivery_failure=True, reflection_policy=REFLECTION_POLICY, artifact_sha256=digest(target), delivery_evidence_sha256={str(p.relative_to(call)): digest(p) for p in evidence})
        save(call / "reflection_metrics.json", metrics)
        return metrics

    @staticmethod
    def validate_reflection_omission(call, target, metrics):
        if metrics.get("reflection_policy") != REFLECTION_POLICY or not target.is_file() or target.is_symlink() or target.read_bytes().strip() or digest(target) != metrics.get("artifact_sha256"):
            raise ProtocolError("Committed reflection delivery outcome changed")
        for name, expected in metrics.get("delivery_evidence_sha256", {}).items():
            path = call / name
            if not path.is_file() or path.is_symlink() or digest(path) != expected:
                raise ProtocolError("Committed reflection delivery evidence changed")

    @staticmethod
    def publish_call(call, directory, step):
        for source in call.glob(f"{step}_*"):
            if source.is_file():
                shutil.copyfile(source, directory / source.name)

    def generation_prompt(self, instance):
        return PromptBuilder.build_generation_prompt((instance / "prompt.txt").read_text(), solver_timeout=self.solver_timeout, solver_threads=self.solver_threads)

    def run_attempt(self, idx, attempt_dir, problem, gold):
        attempt_dir.mkdir(parents=True, exist_ok=True)
        saved = read(attempt_dir / "score_receipt.json")
        if saved:
            for name, expected in saved["artifact_sha256"].items():
                if not (attempt_dir / name).exists() or digest(attempt_dir / name) != expected:
                    raise ProtocolError(f"Scored artifact changed: {name}")
            # The score receipt is committed before summary.json; restore the summary after an interruption between them.
            summary = read(attempt_dir / "summary.json")
            if summary is None:
                save(attempt_dir / "summary.json", saved["result"])
            elif summary != saved["result"]:
                raise ProtocolError("summary.json differs from the committed score receipt")
            return AttemptResult(**saved["result"])
        started = time.perf_counter()
        instance = attempt_dir.parent.parent
        case_identity = read(instance / "identity.json") or {}
        canonical = Path(case_identity.get("canonical_source", instance)) / "data.json"
        if not canonical.is_file() or digest(canonical) != case_identity.get("inputs", {}).get("data.json"):
            raise ProtocolError("Canonical scoring input differs from the frozen case identity")
        context = [instance / "data.json"]
        if self.mode == "reflection" and idx > 1:
            context += previous_attempt_files(attempt_dir.parent, idx)
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
                try:
                    exec_ok, stdout, stderr, output = self.run_solver(attempt_dir, self.solver_timeout, self.solver_threads)
                except SolverLaunchError as error:
                    save(attempt_dir / f"solver_infrastructure_{uuid.uuid4().hex}.json", error.evidence | {"solver_sha256": code_hash})
                    raise InfrastructureError("Solver launcher unavailable; see structured execution diagnostics") from error
                elapsed = time.perf_counter() - execution_started
                native_receipt = read(attempt_dir / "native_solver_receipt.json")
                scored_seconds = native_receipt["result"]["execution_seconds"] if native_receipt else elapsed
                if digest(canonical) != case_identity["inputs"]["data.json"]:
                    raise ProtocolError("Canonical scoring input changed during execution")
                runtime_input = instance / "data.json"
                if not runtime_input.is_file() or runtime_input.is_symlink() or digest(runtime_input) != case_identity["inputs"]["data.json"]:
                    save(attempt_dir / "input_integrity_failure.json", {"expected_sha256": case_identity["inputs"]["data.json"], "observed_sha256": digest(runtime_input) if runtime_input.is_file() and not runtime_input.is_symlink() else None})
                    if runtime_input.exists() or runtime_input.is_symlink():
                        runtime_input.rename(attempt_dir / f"altered_input_{uuid.uuid4().hex}")
                    shutil.copyfile(canonical, runtime_input)
                    exec_ok, output = False, None
                    stderr += "\n[framework] execution input changed; the candidate failed and the canonical input was restored"
                execution = {"exec_ok": exec_ok, "stdout": stdout, "stderr": stderr, "output": output, "execution_seconds": scored_seconds, "solver_sha256": code_hash, "output_sha256": digest(attempt_dir / "output.json") if (attempt_dir / "output.json").exists() else None}
                save(execution_path, execution)
                if not (attempt_dir / "solver_invocation.json").exists():
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
                verification = OutputVerifier.verify_output(self.suite, problem, canonical, attempt_dir / "output.json", self.tol)
                issues.extend(verification.get("issues", []))
                if verification.get("objective_value") is not None:
                    objective = float(verification["objective_value"])
            else:
                issues.append(stderr or "solver execution failed")
            prompt = algorithm_review_prompt(self.suite)
            judged = self.edit(self.judge_client, attempt_dir / "alg_verifier.md", prompt, [attempt_dir / "solver.py"], "llm_verifier")
            notes = (attempt_dir / "alg_verifier.md").read_text().strip()
            verdict = parse_verdict(notes)
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
            if metrics and (metrics.get("ok") or metrics.get("file_delivery_failure")) and not (metrics_path.parent / "rejected.json").exists():
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
        context = [instance / "data.json"] + [attempt_dir / name for name in ("solver.py", "exec.log", "alg_verifier.md", "output.json")]
        reflected = self.edit(self.reflection_client, attempt_dir / "debug.md", prompt, context, "reflection")
        if reflected.get("file_delivery_failure"):
            feedback = attempt_dir / "reflection_feedback.md"
            contents = reflection_delivery_feedback(result.attempt) + "\n"
            if feedback.exists() and feedback.read_text() != contents:
                raise ProtocolError("Framework reflection feedback changed")
            feedback.write_text(contents)
        result.metrics["reflection"] = reflected
        return result

    def prepare(self, source, root, mode):
        destination = root / mode / self.suite / self.problems[0] / source.name
        destination.mkdir(parents=True, exist_ok=True)
        expected = {"protocol_id": self.protocol_id, "reflection_policy": REFLECTION_POLICY, "prompt_policies": {"feedback": FEEDBACK_POLICY, "workspace": WORKSPACE_POLICY}, "model": self.model, "mode": mode, "max_attempts": self.max_attempts, "inputs": {name: digest(source / name) for name in ("data.json", "prompt.txt")}, "references": {name: digest(source / name) for name in ("meta.json", "gold_monolithic.json") if (source / name).exists()}}
        expected["canonical_source"] = str(source.resolve())
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
        from src.experiments.completion import TERMINAL_RECEIPT, TerminalCompletionError, approved_receipts, has_terminal_receipt, load_policy_summary, policy_is_complete
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
                    elif mode == "reflection" and idx < self.max_attempts:
                        # Scoring is already durable even when this separate stage fails.
                        result = self.ensure_reflection(directory, result)
                results.append(result.to_dict())
                summaries[mode] = self.summary(source, destination, mode, results)
                print(f"{self.model} {source.name} {mode} attempt {idx}/{self.max_attempts}: {result.status}", flush=True)
        return summaries
