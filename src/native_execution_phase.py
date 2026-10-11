"""Versioned execution and evaluator amendments around a frozen campaign."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import threading
import time
import uuid


CONCURRENCY_CHANGE = {"active_codex_invocation_limit": {"from": 1, "to": 2}}
RECOVERY_CHANGE = {"generated_solver_file_not_found": "score_failure_if_receipted_solver_and_canonical_inputs_are_intact", "agy_reflection_omission_local_retry_delays_seconds": [5, 10]}
NULL_SOLUTION_CHANGE = {"explicit_null_cg_solution": "reject_csp_gap_shift_null_solution_as_candidate_failure_before_normal_judge"}
GENERATION_FEEDBACK_CHANGE = {"missing_generation_failure_feedback": "add_receipted_framework_diagnostic_only_to_uncommitted_agent_failed_reflections"}
TERMINAL_COMPLETION_CHANGE = "honor_hash_pinned_user_adjudications_without_synthetic_attempts"
USER_PAUSE_GUARD_CHANGE = "require_explicit_resumption_before_run_or_watch"
WORKSPACE_CHANGE = "explicit-workspace-target-v1"
WRONG_TARGET_CHANGE = "fail_future_verified_wrong_target_reflections_without_synthetic_attempts"


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def save(path, value):
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def validate_phase(root, phase_path, *, executing_file=None):
    phase = json.loads(phase_path.read_text())
    payload = dict(phase)
    phase_id = payload.pop("phase_id")
    if identity(payload) != phase_id:
        raise ValueError("Execution phase manifest changed")
    changes = dict(phase["changes"])
    workspace_change = changes.pop("agy_workspace_clarification", None)
    wrong_target_change = changes.pop("wrong_target_reflection_failure", None)
    if workspace_change not in (None, WORKSPACE_CHANGE) or wrong_target_change not in (None, WRONG_TARGET_CHANGE):
        raise ValueError("Unrecognized workspace clarification")
    if workspace_change is not None:
        if wrong_target_change != WRONG_TARGET_CHANGE or not phase.get("workspace_change_authorization"):
            raise ValueError("Workspace clarification needs its explicit failure rule and authorization")
        if digest(phase_path.parent / "agy_client.py") != phase.get("agy_client_sha256"):
            raise ValueError("Clarified AGY adapter changed")
    elif wrong_target_change is not None:
        raise ValueError("Wrong-target failure requires the clarified workspace")
    agy_order = changes.pop("agy_order", None)
    terminal_completion = changes.pop("terminal_policy_completion", None)
    if terminal_completion not in (None, TERMINAL_COMPLETION_CHANGE):
        raise ValueError("Unrecognized terminal-completion amendment")
    pause_guard = changes.pop("user_pause_guard", None)
    if pause_guard not in (None, USER_PAUSE_GUARD_CHANGE):
        raise ValueError("Unrecognized user-pause guard")
    if changes not in (CONCURRENCY_CHANGE, CONCURRENCY_CHANGE | {"runner_recovery": RECOVERY_CHANGE}, CONCURRENCY_CHANGE | {"runner_recovery": RECOVERY_CHANGE | NULL_SOLUTION_CHANGE}, CONCURRENCY_CHANGE | {"runner_recovery": RECOVERY_CHANGE | NULL_SOLUTION_CHANGE | GENERATION_FEEDBACK_CHANGE}):
        raise ValueError("Unrecognized execution-phase change")
    if "runner_recovery" in phase["changes"]:
        runner = phase_path.parent / "paired_runner.py"
        if digest(runner) != phase.get("runner_sha256"):
            raise ValueError("Recovery runner changed")
    if digest(root / "campaign.json") != phase["base_manifest_sha256"]:
        raise ValueError("Base campaign manifest changed")
    base = json.loads((root / "campaign.json").read_text())
    if base["protocol_id"] != phase["base_protocol_id"] or base["workers"] != 2:
        raise ValueError("Base protocol or worker bound differs")
    if agy_order is not None and (agy_order.get("from") != base.get("agy_order") or sorted(agy_order.get("to", [])) != sorted(base.get("agy_order", []))):
        raise ValueError("Execution order must preserve the exact AGY model set")
    wrapper = phase_path.parent / "native_execution_phase.py"
    if Path(executing_file or __file__).resolve() != wrapper.resolve() or digest(wrapper) != phase["wrapper_sha256"]:
        raise ValueError("Use the frozen, hash-verified execution wrapper")
    inventory_path = phase_path.parent / "preserved_artifacts.json"
    if digest(inventory_path) != phase["preserved_inventory_sha256"]:
        raise ValueError("Preserved-artifact inventory changed")
    for name, sha in json.loads(inventory_path.read_text()).items():
        path = root / name
        if not path.is_file() or digest(path) != sha:
            raise ValueError("Preexisting experiment artifact changed: " + name)
    if terminal_completion is not None:
        helper = load_completion_helper(phase_path, phase)
        approvals = phase.get("terminal_policy_receipts", {})
        if not approvals:
            raise ValueError("Terminal policy receipt inventory is empty")
        for relative, expected in approvals.items():
            receipt = root / relative
            if receipt.name != helper.TERMINAL_RECEIPT or not receipt.resolve().is_relative_to(root / "runs") or digest(receipt) != expected:
                raise ValueError("Approved terminal policy receipt changed")
            helper.load_policy_summary(receipt.parent, campaign_root=root, approvals=approvals)
    return phase


def load_completion_helper(phase_path, phase, *, install=False):
    path = phase_path.parent / "policy_completion.py"
    if digest(path) != phase.get("policy_completion_sha256"):
        raise ValueError("Terminal-completion helper changed")
    name = "src.agent.policy_completion" if install else "_frozen_policy_completion_validation"
    specification = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(specification)
    if install:
        sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def ensure_execution_allowed(root, command, phase):
    if command == "verify" or "user_pause_guard" not in phase["changes"]:
        return
    path = root / "pause_request.json"
    pause = json.loads(path.read_text()) if path.exists() else {}
    if pause.get("resume_requires_explicit_user_request"):
        raise ValueError("Campaign is paused by the user; record explicit resumption before starting run or watch")


class NativeCallGate:
    """Two shared-lock slots, or one exclusive legacy lock after an auth failure."""
    def __init__(self, state_directory, phase_directory):
        self.state_directory = state_directory
        self.phase_directory = phase_directory
        self.fallback_path = phase_directory / "auth_fallback.json"
        self.condition = threading.Condition()
        self.active = 0

    @property
    def limit(self):
        return 1 if self.fallback_path.exists() else 2

    def fallback(self, marker, *, role, invocation_id):
        with self.condition:
            if not self.fallback_path.exists():
                save(self.fallback_path, {"recorded_at": now(), "effective_codex_invocation_limit": 1, "reason": marker, "role": role, "invocation_id": invocation_id, "policy": "drain admitted calls, then serialize; do not change credentials"})
            self.condition.notify_all()

    def descriptor(self, name):
        return os.open(self.state_directory / name, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)

    @contextmanager
    def enter(self):
        with self.condition:
            while self.active >= self.limit:
                self.condition.wait(timeout=0.2)
            parallel = self.limit == 2
            self.active += 1
        legacy, slot = None, None
        try:
            legacy = self.descriptor("codex-native-auth.lock")
            # Existing serialized runners use this same lock exclusively.
            fcntl.flock(legacy, fcntl.LOCK_SH if parallel else fcntl.LOCK_EX)
            if parallel:
                while slot is None:
                    for index in range(2):
                        candidate = self.descriptor(f"codex-native-auth.slot{index}.lock")
                        try:
                            fcntl.flock(candidate, fcntl.LOCK_EX | fcntl.LOCK_NB)
                            slot = candidate
                            break
                        except BlockingIOError:
                            os.close(candidate)
                    if slot is None:
                        time.sleep(0.05)
            yield
        finally:
            for descriptor in (slot, legacy):
                if descriptor is not None:
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
                    os.close(descriptor)
            with self.condition:
                self.active -= 1
                self.condition.notify_all()


def auth_failure_marker(result, stderr):
    text = " ".join([result.error or "", *result.warnings, stderr]).lower()
    markers = ("refresh_token_reused", "refresh token", "token refresh failed", "failed to refresh access token", "failed to refresh authentication", "token has already been used", "unauthorized", "authentication failed", "authentication required", "access token has expired", "not logged in", "401 unauthorized", "status code 401", "http 401")
    return next((marker for marker in markers if marker in text), None)


def install_overrides(root, phase_path, phase, *, campaign, paired, codex, agy, state_directory, wrapper_path):
    if "terminal_policy_completion" in phase["changes"]:
        helper = load_completion_helper(phase_path, phase, install=True)
        old_complete = campaign.pair_complete
        approvals = phase["terminal_policy_receipts"]
        def adjudicated_pair_complete(campaign_root, model, case, protocol):
            directories = [campaign_root / "runs" / model / mode / case["suite"] / case["problem"] / case["instance"] for mode in ("plain", "reflection")]
            if not any(helper.has_terminal_receipt(directory) or str((directory / helper.TERMINAL_RECEIPT).relative_to(campaign_root)) in approvals for directory in directories):
                return old_complete(campaign_root, model, case, protocol)
            try:
                summaries = [helper.load_policy_summary(directory, campaign_root=campaign_root, approvals=approvals) for directory in directories]
                return all(helper.policy_is_complete(summary, 10, protocol) for summary in summaries)
            except helper.TerminalCompletionError as error:
                raise paired.ProtocolError(str(error)) from error
        campaign.pair_complete = adjudicated_pair_complete
    """Apply recorded scheduling changes and annotate the active evaluator phase."""
    gate = NativeCallGate(state_directory, phase_path.parent)
    if "agy_workspace_clarification" in phase["changes"]:
        agy.AgyClient.prompt_journal = phase.get("prompt_journey_file")
    if "agy_order" in phase["changes"]:
        original_validate = campaign.validate
        def ordered_validate(campaign_root):
            manifest = original_validate(campaign_root)
            return dict(manifest, agy_order=list(phase["changes"]["agy_order"]["to"]))
        campaign.validate = ordered_validate
    old_lock = paired.runtime_lock
    def lock(name):
        return gate.enter() if name == "codex-native-auth" else old_lock(name)
    paired.runtime_lock = lock
    for client_class, is_codex in ((codex.CodexClient, True), (agy.AgyClient, False)):
        original = client_class.edit_file
        def annotated(self, target, prompt, *, context=None, log_dir=None, step="codex", _original=original, _is_codex=is_codex):
            directory = Path(log_dir or target.parent)
            directory.mkdir(parents=True, exist_ok=True)
            metadata = {"phase_id": phase["phase_id"], "base_protocol_id": phase["base_protocol_id"], "started_at": now(), "role": self.role, "model": self.model, "configured_codex_invocation_limit": 2, "effective_codex_invocation_limit": gate.limit, "controller_pid": os.getpid()}
            save(directory / "execution_phase.json", metadata)
            try:
                result = _original(self, target, prompt, context=context, log_dir=log_dir, step=step)
                stderr_path = directory / f"{step}_stderr.log"
                stderr = stderr_path.read_text(errors="replace") if stderr_path.exists() else ""
                marker = auth_failure_marker(result, stderr) if _is_codex else None
                if marker:
                    gate.fallback(marker, role=self.role, invocation_id=directory.name)
                metadata.update(finished_at=now(), ok=result.ok, auth_fallback_marker=marker)
                save(directory / "execution_phase.json", metadata)
                return result
            except BaseException as error:
                metadata.update(finished_at=now(), exception_type=type(error).__name__)
                save(directory / "execution_phase.json", metadata)
                raise
        client_class.edit_file = annotated
    child_command = [sys.executable, "-u", str(wrapper_path), "run", "--root", str(root), "--phase", str(phase_path)]
    old_save = campaign.save
    def annotated_save(path, value):
        if path in (root / "status.json", root / "watchdog.json"):
            value = dict(value, execution_phase_id=phase["phase_id"], execution_phase_file=str(phase_path), active_codex_invocation_limit=gate.limit)
            if "agy_order" in phase["changes"]:
                value["execution_agy_order"] = phase["changes"]["agy_order"]["to"]
            if path == root / "watchdog.json":
                value["command"] = child_command
        return old_save(path, value)
    campaign.save = annotated_save
    old_supervise = campaign.supervise
    def supervised_phase(command, monitor_root, **kwargs):
        return old_supervise(child_command, monitor_root, **kwargs)
    campaign.supervise = supervised_phase
    return gate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["run", "watch", "verify"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--phase", type=Path, required=True)
    args = parser.parse_args()
    root, phase_path = args.root.resolve(), args.phase.resolve()
    phase = validate_phase(root, phase_path)
    ensure_execution_allowed(root, args.command, phase)
    sys.path.insert(0, str(root / "controller"))
    if "agy_workspace_clarification" in phase["changes"]:
        specification = importlib.util.spec_from_file_location("src.agent.agy_client", phase_path.parent / "agy_client.py")
        module = importlib.util.module_from_spec(specification)
        sys.modules[specification.name] = module
        specification.loader.exec_module(module)
    if "terminal_policy_completion" in phase["changes"]:
        load_completion_helper(phase_path, phase, install=True)
    if "runner_recovery" in phase["changes"]:
        specification = importlib.util.spec_from_file_location("src.agent.paired_runner", phase_path.parent / "paired_runner.py")
        module = importlib.util.module_from_spec(specification)
        sys.modules[specification.name] = module
        specification.loader.exec_module(module)
    from src import native_campaign as campaign
    from src.agent import agy_client as agy, codex_client as codex, paired_runner as paired
    from src.agent.runtime_state import state_root
    if Path(campaign.__file__).resolve() != root / "controller/src/native_campaign.py":
        raise ValueError("Base campaign module is not from its frozen controller")
    campaign.validate(root)
    if args.command == "verify":
        print(json.dumps({"validated": True, "phase_id": phase["phase_id"], "paired_runner": str(Path(paired.__file__).resolve()), "agy_client": str(Path(agy.__file__).resolve()), "campaign": str(Path(campaign.__file__).resolve())}))
        return
    install_overrides(root, phase_path, phase, campaign=campaign, paired=paired, codex=codex, agy=agy, state_directory=state_root(), wrapper_path=Path(__file__).resolve())
    raise SystemExit(campaign.run(root) if args.command == "run" else campaign.watch(root))


if __name__ == "__main__":
    main()
