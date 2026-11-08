"""Resumable native paired retries with a single shared first evaluation.

The paired lifecycle (a shared first attempt, then plain and reflection retries with durable receipts) is independent
of what the agent reads, writes and runs, and of how a candidate is scored:
- ContextSpec (src.experiments.context_spec): the read-only files of each call (instance data representation, history
  sets, transport and mount layout);
- DeliverySpec and RuntimeSpec (src.agent.specs): the writable/collected targets and the run-time contract;
- ExecutionPolicy (src.experiments.execution_policy): executes the delivered candidate and evaluates it; every stop
  and feedback decision reads its Evaluation;
- the Part 2 section adapter (src.agent.prompts.mounted, PROSE_SECTIONS by default) and the judge's audit extensions
  (src.agent.prompts.judge).
The defaults are the paper's and the small-scale protocol and keep every record of the earlier fixed implementation;
spec_record() names any other spec in the case identity and in every request fingerprint.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
import hashlib
from pathlib import Path
import re
import shutil
import threading
import time
from typing import Any
import uuid

from src.agent.prompts.builder import PromptBuilder
from src.agent.prompts.judge import PAPER_AUDIT_TARGET, algorithm_review_prompt, judge_policy, parse_verdict
from src.agent.mounted_context import manifest as mounted_manifest, mounted_files
from src.agent.prompts.method import DECOMPOSITION_POLICY, require_contract
from src.agent.prompts.mounted import PROSE_SECTIONS, review_files
from src.agent.prompts.policy import FEEDBACK_POLICY, WORKSPACE_POLICY, REFLECTION_POLICY, reflection_delivery_feedback
from src.agent.specs import DEFAULT_RUNTIME, MOUNT_LAYOUT, SINGLE_FILE_DELIVERY, RuntimeSpec
from src.experiments.context_spec import LEGACY_CONTEXTS, PAPER_HISTORY, context_spec, judge_delivery
from src.experiments.data_context import DATA_CONTEXT_POLICIES
from src.experiments.environment import require_runtime_environment
from src.experiments.exec_log import ExecLogger
from src.experiments.execution_policy import SMALL_SCALE_NATIVE_POLICY, AttemptCase, Evaluation, SmallScaleNativePolicy
from src.experiments.locks import runtime_lock
from src.utils.errors import InfrastructureError, ProtocolError
from src.utils.families import problems
from src.utils.records import digest, identity, read, save


# Without an explicit problem the runner keeps its historical default: the paper families of the suite, in this order.
DEFAULT_PROBLEMS = {"bd": ["tsso", "fctp", "km", "lrp", "ufl"], "cg": ["bp", "csp", "gap", "shift"]}
NESTED_NAME = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]*")  # a declared attempt subdirectory: one plain name


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


def clone_or_copy(source: Path, target: Path) -> str:
    """Create `target` (which must not exist) as an APFS copy-on-write clone of `source` (clonefile(2), as `cp -c`):
    an independent file, never a link, that shares data blocks until one side changes. Where the filesystem or the
    platform cannot clone, a plain copy. Returns "clone" or "copy"; the caller verifies the target's sha256."""
    if target.exists() or target.is_symlink():
        raise ProtocolError(f"Refusing to replace an existing file: {target}")
    try:
        import ctypes
        import os
        clonefile = ctypes.CDLL(None, use_errno=True).clonefile
        if clonefile(os.fsencode(source), os.fsencode(target), 0) == 0:
            return "clone"
    except (AttributeError, OSError):
        pass
    if target.exists() or target.is_symlink():
        raise ProtocolError(f"A failed clone left a file behind: {target}")
    shutil.copyfile(source, target)
    return "copy"


def previous_attempt_files(attempts_root: Path, current_idx: int) -> list[Path]:
    """Read-only files of earlier attempts given to a reflection-policy generation (the paper's history sets)."""
    return PAPER_HISTORY.previous_attempts(attempts_root, current_idx)


class PairedRunner:
    """Run one (model, instance) pair: a shared first attempt, then plain and reflection retries."""

    def __init__(self, *, client, judge_client, solver_executor, reflection_client=None, protocol_id="native-qualitative-v2", on_stage=None, call_lock=None,
                 model=None, suite="bd", problem=None, max_attempts=10, solver_timeout=60, agent_timeout=900, solver_threads=1,
                 solver_cooldown_seconds=0, tol=1e-6, gap_threshold=0.01, data_context=None, decomposition_contract=DECOMPOSITION_POLICY, context_delivery=None,
                 context=None, delivery=None, runtime_spec=None, execution_policy=None, task_sections=None, audit_extensions=(), audit_target=None,
                 judge_context_delivery=None):
        if suite not in DEFAULT_PROBLEMS:
            raise ValueError("Choose suite bd or cg.")
        if max_attempts < 1 or solver_threads < 1 or min(solver_timeout, agent_timeout, tol) <= 0 or gap_threshold < 0 or solver_cooldown_seconds < 0:
            raise ValueError("Attempt/time/tolerance budgets must be positive; gap threshold must be nonnegative.")
        if data_context not in DATA_CONTEXT_POLICIES:
            raise ValueError(f"Unknown data-context policy: {data_context}")
        # The method judge applies the campaign's contract, the same one rendered into the instance prompt.txt.
        self.decomposition_contract = require_contract(decomposition_contract)
        # What each call reads (src.experiments.context_spec). The recorded settings select a reviewed spec: None/None
        # keeps the paper's pasted full data.json, data_context the structural preview, context_delivery
        # CONTEXT_DELIVERY_MOUNTED the read-only files in the agent workspace with the structured prompt
        # (src.agent.prompts.mounted). Any other spec is given explicitly as `context`. The judge's context (the
        # sources under review) stays pasted in every spec.
        if context is None:
            context = context_spec(data_context, context_delivery)
        elif data_context is not None or context_delivery is not None:
            raise ValueError("Give either a ContextSpec or the recorded data_context/context_delivery settings")
        self.context = context
        self.data_context, self.context_delivery = context.data_context, context.context_delivery
        # The method judge's context (prompt_protocol.judge_context_delivery): None pastes the sources under review (the
        # paper); the mounted setting resolves to the client transport and REVIEW_LAYOUT (context_spec.judge_delivery).
        self.judge_context_delivery = judge_context_delivery
        self.judge_transport, self.judge_layout = judge_delivery(judge_context_delivery, context)
        # The controller-side environment check (the local Python/Gurobi versions for the native policy), in its earlier place.
        self.runtime_environment = require_runtime_environment() if execution_policy is None else execution_policy.runtime_environment()
        self.model, self.suite = model, suite
        self.problems = list(problem) if problem else list(DEFAULT_PROBLEMS[suite])
        if any(p not in problems(suite) for p in self.problems):
            raise ValueError(f"Problems must be registered {suite} families: {problems(suite)}")
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
        # What the agent writes and the run-time contract it is told (src.agent.specs); the output readback policy and
        # the memory budget are the executor's (NativeSandboxExecutor.output_policy, .memory_limit_mib; None for the paper).
        self.delivery = delivery or SINGLE_FILE_DELIVERY
        memory = getattr(solver_executor, "memory_limit_mib", None)
        self.runtime_spec = runtime_spec or RuntimeSpec(solver_timeout, solver_threads, getattr(solver_executor, "output_policy", None), memory)
        if (self.runtime_spec.timeout, self.runtime_spec.threads) != (solver_timeout, solver_threads):
            raise ValueError("The runtime contract's budget differs from the solver budget")
        if self.runtime_spec.memory_mib != memory:
            raise ValueError("The runtime contract's memory budget differs from the executor's")
        # How a candidate runs and is scored; the paper's native policy calls run_solver (also when a test replaces it).
        # The executor's evidence schema (sandbox.EVIDENCE_SCHEMAS; None for the paper and for plain callables) decides
        # how execution.json stores the output and whether the shared first attempt's output.json is cloned.
        self.evidence_schema = getattr(solver_executor, "evidence_schema", None)
        self.policy = execution_policy or SmallScaleNativePolicy(executor=lambda *args: self.run_solver(*args), suite=suite, tol=tol, gap_threshold=gap_threshold,
                                                                 cooldown_seconds=solver_cooldown_seconds,
                                                                 **({"evidence_schema": self.evidence_schema, "output_limit": solver_executor.output_limit} if self.evidence_schema else {}))
        self.policy.check(delivery=self.delivery, runtime=self.runtime_spec)
        if memory is not None and self.policy.id == SMALL_SCALE_NATIVE_POLICY:
            raise ValueError("The paper's execution policy has no memory budget; a budgeted executor needs its reviewed variant policy")
        if any(not isinstance(name, str) or not NESTED_NAME.fullmatch(name) for name in (*self.delivery.nested, *self.policy.nested)):
            raise ValueError("Declared attempt subdirectories must be plain names")
        self.task_sections = task_sections or PROSE_SECTIONS
        self.audit_extensions = tuple(audit_extensions)
        self.audit_target = audit_target or PAPER_AUDIT_TARGET
        # A pasted context is the paper's: its prompt.txt prose, the runtime note and no author-side audit rule. Other
        # sections, runtime contracts and shared method rules need the structured (mounted) prompt.
        if not self.context.mounted and (self.task_sections.id != PROSE_SECTIONS.id or self.runtime_spec.contract != DEFAULT_RUNTIME.contract or any(e.author for e in self.audit_extensions)
                                         or self.runtime_spec.memory_mib is not None):
            raise ValueError("A pasted context renders the paper's prompt.txt and runtime note; this section adapter, runtime contract, memory budget or method rule needs the mounted-files prompt")

    def spec_record(self) -> dict:
        """The ids of every non-default spec of this runner, recorded in the case identity and in each request (empty
        for the paper and small-scale protocol, whose records therefore stay those of the earlier fixed code)."""
        record = {}
        if self.context not in LEGACY_CONTEXTS.values():
            record["context"] = self.context.id
        if self.delivery != SINGLE_FILE_DELIVERY:
            record["delivery"] = self.delivery.id
        if self.runtime_spec.contract != DEFAULT_RUNTIME.contract:
            record["runtime"] = self.runtime_spec.id
        if self.policy.id != SMALL_SCALE_NATIVE_POLICY:
            record["execution_policy"] = self.policy.id
        if self.task_sections.id != PROSE_SECTIONS.id:
            record["task_sections"] = self.task_sections.id
        if self.audit_extensions or self.audit_target != PAPER_AUDIT_TARGET:
            record["algorithm_judge"] = judge_policy(self.audit_extensions, self.audit_target)
        return record

    def evaluation(self, result: AttemptResult) -> Evaluation:
        """The Evaluation behind every stop and feedback decision: the policy's, restored from the durable record. A
        generation that delivered no solver was never executed or evaluated: it solves nothing and gives no objective
        feedback (the paper's rule gives the same for that record)."""
        if result.status == "agent_failed":
            return Evaluation(status=result.status, feasible=False, verified=False, objective=None, reference_objective=result.gold_objective, gap=None,
                              termination="not_executed", success={}, headline="", issues=result.issues)
        return self.policy.evaluation(result)

    def objective_quality_failed(self, result: AttemptResult) -> bool:
        """Expose one qualitative check outcome, without passing the reference to the prompt (the policy's Evaluation)."""
        return self.evaluation(result).objective_target_missed

    @staticmethod
    def can_early_stop(result: AttemptResult, gap_threshold: float) -> bool:
        """The paper's stop rule on a record (SmallScaleNativePolicy applies it as Evaluation.solved)."""
        return result.status == "ok" and result.exec_ok and result.verify_ok and result.gap_to_gold is not None and result.gap_to_gold <= gap_threshold

    def check_client(self, client, step):
        """A generation or reflection client must place, describe and collect files as this runner's specs do; a mounted
        judge client must use the review layout."""
        if step == "llm_verifier" and self.judge_transport is not None and getattr(client, "context_layout", None) != self.judge_layout:
            raise ProtocolError("The judge client's mounted layout differs from the campaign's")
        mounted = step in ("agent", "reflection") and self.context_delivery is not None
        runtime = getattr(client, "runtime_spec", DEFAULT_RUNTIME)
        if mounted and (getattr(client, "context_layout", MOUNT_LAYOUT) != self.context.layout or runtime.contract != self.runtime_spec.contract or runtime.memory_mib != self.runtime_spec.memory_mib):
            raise ProtocolError(f"The {step} client's mounted layout or runtime contract differs from the campaign's")
        if step == "agent" and getattr(client, "delivery", SINGLE_FILE_DELIVERY) != self.delivery:
            raise ProtocolError(f"The {step} client's delivery differs from the campaign's")

    def edit(self, client, target, prompt, context, step, data_context=None):
        """Commit each physical call once; retain all failed-call evidence."""
        directory = target.parent
        request = {"protocol": self.protocol_id, "reflection_policy": REFLECTION_POLICY, "prompt_policies": {"feedback": FEEDBACK_POLICY, "workspace": WORKSPACE_POLICY}, "step": step, "prompt": prompt, "target": target.name, "context": [{"label": f"{p.parent.name}/{p.name}", "sha256": digest(p)} for p in context if p.exists()]}
        if data_context is not None:
            # Recorded only under an explicit data-context setting, so legacy request fingerprints are unchanged.
            request["data_context"] = data_context
        judge = step == "llm_verifier"
        delivery = self.judge_transport if judge else self.context_delivery if step in ("agent", "reflection") else None
        if getattr(client, "context_delivery", None) != delivery:
            raise ProtocolError(f"The {step} client's context delivery differs from the campaign's")
        self.check_client(client, step)
        if delivery is not None:
            # The mounted layout (path and sha256 of every mounted file) is part of the request fingerprint.
            layout = self.judge_layout if judge else self.context.layout
            # A mounted review is checked here, before any receipt or recovered call is reused: every source exists,
            # one attempt, solver.py first, flat C++ names (src.agent.prompts.mounted.review_files).
            files = review_files(context, layout) if judge else mounted_files(context, layout)
            request["context_delivery"] = {"policy": delivery, "files": [{key: entry[key] for key in ("path", "sha256")} for entry in mounted_manifest(files)]}
            if judge:
                request["context_delivery"] |= {"judge": self.judge_context_delivery, "layout": layout.id}
        specs = self.spec_record()
        if specs:
            request["specs"] = specs
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

    def task_part_sections(self, instance):
        """The Part 2 sections of a structured prompt from the runner's section adapter (the stored prompt.txt by default).
        A method rule of the audit (AuditExtension.author) must be stated to the author as well."""
        sections = self.task_sections(instance, suite=self.suite, problem=self.problems[0], contract=self.decomposition_contract)
        missing = [extension.id for extension in self.audit_extensions if extension.author and not all(text in sections["Method requirements"] for text in extension.paragraphs)]
        if missing:
            raise ProtocolError(f"The author's Method requirements lack the audit's method rules: {missing}")
        return sections

    def generation_prompt(self, instance):
        if getattr(self, "context_delivery", None) is not None:  # tests also call this with a bare namespace as self
            return PromptBuilder.structured_generation(self.task_part_sections(instance), runtime=self.runtime_spec)
        return PromptBuilder.build_generation_prompt((instance / "prompt.txt").read_text(), solver_timeout=self.solver_timeout, solver_threads=self.solver_threads)

    def reflection_prompt(self, instance, objective_quality_failed):
        if self.context_delivery is not None:
            return PromptBuilder.structured_reflection(self.task_part_sections(instance), runtime=self.runtime_spec, objective_quality_failed=objective_quality_failed)
        return PromptBuilder.build_reflection_prompt(self.generation_prompt(instance), objective_quality_failed=objective_quality_failed)

    def run_attempt(self, idx, attempt_dir, problem, gold):
        attempt_dir.mkdir(parents=True, exist_ok=True)
        saved = read(attempt_dir / "score_receipt.json")
        if saved:
            for name, expected in saved["artifact_sha256"].items():
                if not (attempt_dir / name).exists() or digest(attempt_dir / name) != expected:
                    raise ProtocolError(f"Scored artifact changed: {name}")
            if set(self.nested_artifacts(attempt_dir)) != {name for name in saved["artifact_sha256"] if "/" in name}:
                raise ProtocolError("The scored attempt's declared nested artifacts changed")
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
        context, data_record = self.context.generation(instance, attempt_dir.parent, idx, self.mode)
        if (self.mode != "reflection" or idx == 1) and any(Path(path).resolve().is_relative_to(attempt_dir.parent.resolve()) for path in context):
            raise ProtocolError("A first attempt or a plain regeneration receives no file of an earlier attempt")
        generation = self.edit(self.client, attempt_dir / "solver.py", self.generation_prompt(instance), context, "agent", data_context=data_record)
        metrics, queued = {"generation": generation}, 0.0
        if not generation["ok"]:
            result = AttemptResult(idx, "agent_failed", False, False, None, self.policy.reference_objective(gold), None, [generation["error"] or "Model omitted the requested solver"], metrics=metrics)
        else:
            case = AttemptCase(attempt_dir, instance, canonical, case_identity["inputs"]["data.json"], problem)
            execution = self.policy.execute(case, delivery=self.delivery, runtime=self.runtime_spec, stage=lambda phase: self.on_stage(mode=self.mode, attempt=idx, phase=phase))
            metrics.update(execution.metrics)
            queued = execution.queue_seconds

            def audit():
                """The method judge (pasted source review of the delivered sources); the policy decides when it runs."""
                reviewed = self.delivery.reviewed_sources(attempt_dir, attempt_dir / "calls" / generation["invocation_id"])
                prompt = algorithm_review_prompt(self.suite, self.decomposition_contract, sources=[path.relative_to(attempt_dir).as_posix() for path in reviewed], extensions=self.audit_extensions,
                                                 target=self.audit_target)
                judged = self.edit(self.judge_client, attempt_dir / "alg_verifier.md", prompt, reviewed, "llm_verifier")
                notes = (attempt_dir / "alg_verifier.md").read_text().strip()
                verdict = parse_verdict(notes)
                if verdict is None:
                    save(attempt_dir / "calls" / judged["invocation_id"] / "rejected.json", {"reason": "malformed algorithm verdict"})
                    (attempt_dir / "llm_verifier_receipt.json").unlink()
                    raise InfrastructureError("Algorithm judge produced a malformed verdict")
                metrics["algorithm_verification"] = judged
                return verdict, notes

            evaluation = self.policy.evaluate(execution, gold, case=case, audit=audit)
            if evaluation.infrastructure:
                # Never scored or charged to the agent: the attempt is evaluated again after repair.
                raise InfrastructureError("Evaluation infrastructure failed: " + "; ".join(evaluation.infrastructure))
            if evaluation.record:
                metrics["evaluation"] = dict(evaluation.record)
            result = AttemptResult(idx, evaluation.status, execution.ok, evaluation.verified, evaluation.objective, evaluation.reference_objective, evaluation.gap, evaluation.issues,
                                   evaluation.method_verdict, evaluation.method_notes, stdout=execution.stdout, stderr=execution.stderr, feasibility_ok=evaluation.feasible, metrics=metrics)
            lines = ExecLogger.build_exec_lines(execution.stdout, execution.stderr)
            ExecLogger.append_output_message(lines, evaluation.message)
            ExecLogger.append_verification_issues(lines, evaluation.issues)
            ExecLogger.append_llm_verifier(lines, evaluation.method_notes)
            ExecLogger.write_exec_log(attempt_dir, lines)
        metrics["total_seconds"] = time.perf_counter() - started
        if queued:  # this call's solver-lock queue time is not attempt time (the lanes policy records it as solver_queue_seconds)
            metrics["total_seconds"] = max(0.0, metrics["total_seconds"] - queued)
        artifacts = {name: digest(attempt_dir / name) for name in ("solver.py", "output.json", "alg_verifier.md", "exec.log") if (attempt_dir / name).exists()}
        artifacts.update(self.nested_artifacts(attempt_dir))  # declared nested sources and evidence (none in the paper layout)
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
        prompt = self.reflection_prompt(instance, self.evaluation(result).objective_target_missed)
        context, data_record = self.context.reflection(instance, attempt_dir)
        reflected = self.edit(self.reflection_client, attempt_dir / "debug.md", prompt, context, "reflection", data_context=data_record)
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
        # The frozen inputs (data.json and what the section adapter reads) and the policy's reference files.
        inputs = ("data.json", *self.task_sections.inputs)
        expected = {"protocol_id": self.protocol_id, "reflection_policy": REFLECTION_POLICY, "prompt_policies": {"feedback": FEEDBACK_POLICY, "workspace": WORKSPACE_POLICY}, "model": self.model, "mode": mode, "max_attempts": self.max_attempts, "inputs": {name: digest(source / name) for name in inputs}, "references": {name: digest(source / name) for name in self.policy.reference_files if (source / name).exists()}}
        expected["canonical_source"] = str(source.resolve())
        if self.data_context is not None:
            expected["data_context"] = self.data_context
        if self.decomposition_contract != DECOMPOSITION_POLICY:
            expected["decomposition_contract"] = self.decomposition_contract
        if self.context_delivery is not None:
            expected["context_delivery"] = self.context_delivery
        if self.judge_context_delivery is not None:
            expected["judge_context_delivery"] = self.judge_context_delivery
        specs = self.spec_record()
        if specs:
            expected["specs"] = specs
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

    SHARED_PROTECTED = frozenset({"solver.py", "output.json", "alg_verifier.md"})

    def nested_artifacts(self, attempt_dir: Path) -> dict[str, str]:
        """{relative path: sha256} of every file under the attempt subdirectories that the delivery and the policy
        declare as part of the attempt (DeliverySpec.nested, ExecutionPolicy.nested; none in the paper layout). Only
        real directories and regular files are allowed there."""
        manifest = {}
        for name in dict.fromkeys(tuple(self.delivery.nested) + tuple(self.policy.nested)):
            root = attempt_dir / name
            if root.is_symlink() or (root.exists() and not root.is_dir()):
                raise ProtocolError(f"Attempt directory {name}/ is not a real directory")
            for path in sorted(root.rglob("*")) if root.is_dir() else []:
                if path.is_symlink() or not (path.is_file() or path.is_dir()):
                    raise ProtocolError(f"Unsafe entry in the attempt directory {name}/: {path.relative_to(attempt_dir)}")
                if path.is_file():
                    manifest[path.relative_to(attempt_dir).as_posix()] = digest(path)
        return manifest

    def shared_artifacts(self, first_dir: Path) -> dict[str, str]:
        """The manifest {relative path: sha256} of the shared first attempt's artifact set: its top-level files except
        identity.json (in directory order), then the nested files committed in its score receipt, which must be
        exactly the nested files present."""
        manifest = {path.name: digest(path) for path in first_dir.iterdir() if path.is_file() and path.name not in {"identity.json"}}
        committed = {name: sha for name, sha in ((read(first_dir / "score_receipt.json") or {}).get("artifact_sha256") or {}).items() if "/" in name}
        if self.nested_artifacts(first_dir) != committed:
            raise ProtocolError("The shared first attempt's nested artifacts differ from its score receipt")
        manifest.update(committed)
        return manifest

    def copy_shared_first(self, first_dir: Path, reflection_first: Path):
        """Project the shared first attempt into the reflection policy (copies, never links); a committed artifact
        that already exists there must be identical, and the projection's declared nested files must end up exactly
        the committed ones (no extra, linked or special entry)."""
        manifest = self.shared_artifacts(first_dir)
        committed = {name: sha for name, sha in manifest.items() if "/" in name}
        if any(committed.get(name) != sha for name, sha in self.nested_artifacts(reflection_first).items()):
            raise ProtocolError("Shared first-attempt projection differs")
        for name, sha in manifest.items():
            source, target = first_dir / name, reflection_first / name
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                if self.evidence_schema is not None and name == "output.json":
                    clone_or_copy(source, target)  # the one large file: an independent copy-on-write clone where possible
                else:
                    shutil.copyfile(source, target)
                if digest(target) != sha:
                    raise ProtocolError("Shared first-attempt copy differs from its manifest")
            elif (name in self.SHARED_PROTECTED or "/" in name) and sha != digest(target):
                raise ProtocolError("Shared first-attempt projection differs")
        if self.nested_artifacts(reflection_first) != committed:
            raise ProtocolError("Shared first-attempt projection differs")

    def summary(self, source, destination, mode, results):
        value = {"instance": source.name, "suite": self.suite, "problem": self.problems[0], "source": str(source), "output_dir": str(destination), "backend": self.client.protocol().get("backend"), "model": self.model, "reflection": mode == "reflection", "solver_threads": self.solver_threads, "solver_timeout": self.solver_timeout, "solver_cooldown_seconds": self.solver_cooldown_seconds, "max_attempts": self.max_attempts, "gap_threshold": self.gap_threshold, "protocol_id": self.protocol_id, "shared_first_attempt": True, "complete": len(results) == self.max_attempts, "attempts": results}
        save(destination / "instance_summary.json", value)
        return value

    def run_pair(self, source, root):
        """The caller owns this (model, case) pair; policies never run concurrently."""
        from src.experiments.completion import TERMINAL_RECEIPT, TerminalCompletionError, approved_receipts, case_settings, has_terminal_receipt, load_policy_summary, policy_is_complete
        saved = {mode: root / mode / self.suite / self.problems[0] / source.name for mode in ("plain", "reflection")}
        campaign_root = root.parent.parent if root.parent.name == "runs" and (root.parent.parent / "campaign.json").exists() else None
        try:
            approvals = approved_receipts(campaign_root) if campaign_root else {}
        except TerminalCompletionError as error:
            raise ProtocolError(str(error)) from error
        if any(has_terminal_receipt(directory) or (campaign_root and str((directory / TERMINAL_RECEIPT).relative_to(campaign_root)) in approvals) for directory in saved.values()):
            try:
                summaries = {mode: load_policy_summary(directory, campaign_root=campaign_root, approvals=approvals) for mode, directory in saved.items()}
                # A closed pair is reused only under the contract, deliveries and specs it was judged with (an absent
                # field is the paper's), the predicate of the campaign's own completion checks.
                expected = case_settings({"decomposition_contract": self.decomposition_contract, "context_delivery": self.context_delivery,
                                          "judge_context_delivery": self.judge_context_delivery}, self.spec_record())
                if any(case_settings(read(directory / "identity.json") or {}) != expected for directory in saved.values()):
                    raise ProtocolError("Case protocol changed")
                if all(policy_is_complete(summary, self.max_attempts, self.protocol_id) for summary in summaries.values()):
                    return summaries
                raise ProtocolError("An adjudicated pair is not fully closed")
            except TerminalCompletionError as error:
                raise ProtocolError(str(error)) from error
        destinations = {mode: self.prepare(source, root, mode) for mode in ("plain", "reflection")}
        gold = self.policy.references(source)
        self.mode = "plain"
        first_dir = destinations["plain"] / "attempts/attempt_01"
        first = self.run_attempt(1, first_dir, self.problems[0], gold)
        shared_id = identity({"protocol": self.protocol_id, "model": self.model, "instance": source.name, "attempt": 1})
        # Copies keep reflection notes out of the independent policy and make the old report layout usable.
        reflection_first = destinations["reflection"] / "attempts/attempt_01"
        reflection_first.mkdir(exist_ok=True)
        self.copy_shared_first(first_dir, reflection_first)
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
                    reused = self.policy.reused(successful)
                    result = AttemptResult(idx, "skipped_previous_solved", reused["exec_ok"], reused["verify_ok"], reused["objective"], self.policy.reference_objective(gold), reused["gap_to_gold"], [],
                                           reused_from_previous=True, first_success_attempt=first_success, feasibility_ok=reused["feasibility_ok"], metrics=reused["metrics"])
                else:
                    result = deepcopy(first) if idx == 1 else self.run_attempt(idx, directory, self.problems[0], gold)
                    # The stop rule is the policy's: success of its headline target (the paper: within the gap of the gold).
                    if self.evaluation(result).solved:
                        successful, first_success = result, idx
                    elif mode == "reflection" and idx < self.max_attempts:
                        # Scoring is already durable even when this separate stage fails.
                        result = self.ensure_reflection(directory, result)
                results.append(result.to_dict())
                summaries[mode] = self.summary(source, destination, mode, results)
                print(f"{self.model} {source.name} {mode} attempt {idx}/{self.max_attempts}: {result.status}", flush=True)
        return summaries
