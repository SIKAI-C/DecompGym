"""Validate explicitly adjudicated policy failures without rewriting experiment data."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path


TERMINAL_RECEIPT = "terminal_failure.json"
TERMINAL_KIND = "user_adjudicated_terminal_failure"
TERMINAL_AMENDMENT = "honor_hash_pinned_user_adjudications_without_synthetic_attempts"
DELIVERY_RECEIPT = "reflection_delivery_failure.json"
DELIVERY_KIND = "terminal_reflection_delivery_failure"
DELIVERY_AMENDMENT = "fail_future_verified_wrong_target_reflections_without_synthetic_attempts"
WORKSPACE_AMENDMENT = "explicit-workspace-target-v1"


def is_terminal_failure(summary: dict) -> bool:
    return summary.get("completion_kind") in (TERMINAL_KIND, DELIVERY_KIND)


def has_terminal_receipt(directory: Path) -> bool:
    return any((directory / name).is_file() for name in (TERMINAL_RECEIPT, DELIVERY_RECEIPT))


class TerminalCompletionError(ValueError):
    """A claimed terminal outcome lacks intact, explicitly approved evidence."""


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identity(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def read(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as error:
        raise TerminalCompletionError(f"Cannot read completion evidence: {path}") from error


def require(condition, message):
    if not condition:
        raise TerminalCompletionError(message)


def _evidence_path(root: Path, relative: str) -> Path:
    path = root / relative
    require(not Path(relative).is_absolute() and ".." not in Path(relative).parts,
            "Completion evidence path is not relative")
    require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(root.resolve()),
            f"Completion evidence is missing or unsafe: {relative}")
    return path


def campaign_root_for(directory: Path) -> Path:
    for parent in directory.resolve().parents:
        if (parent / "campaign.json").is_file():
            return parent
    raise TerminalCompletionError("Terminal policy has no campaign manifest")


def approved_receipts(root: Path) -> dict:
    if not (root / "active_execution_phase.json").exists():
        return {}
    pointer = read(root / "active_execution_phase.json")
    phase_path = Path(pointer["phase_file"])
    if not phase_path.is_absolute():
        phase_path = root / phase_path
    require(phase_path.resolve().is_relative_to((root / "execution_phases").resolve()),
            "Terminal policy approval is outside the campaign phases")
    phase = read(phase_path)
    if "terminal_policy_completion" not in phase.get("changes", {}):
        return {}
    payload = {key: value for key, value in phase.items() if key != "phase_id"}
    require(identity(payload) == phase.get("phase_id") == pointer.get("phase_id"),
            "Terminal policy approval phase changed")
    require(digest(root / "campaign.json") == phase.get("base_manifest_sha256"),
            "Terminal policy campaign manifest changed")
    require(phase.get("changes", {}).get("terminal_policy_completion") == TERMINAL_AMENDMENT,
            "Active phase does not authorize terminal adjudications")
    approvals = phase.get("terminal_policy_receipts")
    require(isinstance(approvals, dict) and bool(approvals), "Terminal approval inventory is missing")
    return approvals


def policy_is_complete(summary: dict, max_attempts: int, protocol: str | None = None) -> bool:
    if not summary or not summary.get("complete"):
        return False
    if protocol is not None:
        require(summary.get("protocol_id") == protocol and summary.get("max_attempts") == max_attempts,
                "Completed policy has an incompatible protocol")
    if is_terminal_failure(summary):
        require(summary.get("terminal_receipt_sha256") and summary.get("outcome") == "failed",
                "Terminal policy was not validated")
        return True
    return len(summary.get("attempts", [])) == max_attempts


def _validate(directory: Path, root: Path, raw: dict, receipt: dict):
    manifest = read(root / "campaign.json")
    payload = {key: value for key, value in receipt.items() if key != "receipt_id"}
    require(identity(payload) == receipt.get("receipt_id"), "Terminal receipt identity changed")
    require(receipt.get("schema_version") == 1 and receipt.get("kind") == TERMINAL_KIND,
            "Unsupported terminal adjudication")
    require(receipt.get("outcome") == "failed" and receipt.get("stage") == "reflection"
            and receipt.get("mode") == "reflection", "Terminal adjudication is not a reflection failure")
    require(receipt.get("authorization", {}).get("scope") == "one_saved_policy"
            and bool(receipt["authorization"].get("user_decision")), "Explicit user decision is missing")
    expected_directory = root / "runs" / receipt["model"] / receipt["mode"] / receipt["suite"] / receipt["problem"] / receipt["instance"]
    require(directory.resolve() == expected_directory.resolve(), "Terminal policy identity/path differs")
    for key in ("model", "suite", "problem", "instance", "max_attempts", "protocol_id"):
        require(raw.get(key) == receipt.get(key), f"Terminal policy identity differs: {key}")
    require(raw.get("reflection") is True and not raw.get("complete"),
            "Adjudication must retain an unfinished reflection summary")
    require(receipt["protocol_id"] == manifest["protocol_id"] and receipt["max_attempts"] == manifest["max_attempts"],
            "Terminal policy budget or campaign protocol differs")
    case = next((case for case in manifest["instances"] if all(case[key] == receipt[key] for key in ("suite", "problem", "instance"))), None)
    require(case is not None and case["sha256"] == receipt.get("canonical_input_sha256"),
            "Terminal policy is not in the fixed campaign inputs")
    for name, expected in case["sha256"].items():
        source = root / "dataset" / case["suite"] / case["problem"] / case["instance"] / name
        require(digest(source) == expected, f"Canonical terminal-policy input changed: {name}")
    evidence = receipt.get("evidence_sha256", {})
    require(all(name in evidence for name in ("instance_summary.json", "identity.json", "shared_first.json", "data.json", "prompt.txt")),
            "Terminal policy identity/input evidence is incomplete")
    for name, expected in evidence.items():
        require(digest(_evidence_path(directory, name)) == expected, f"Terminal policy evidence changed: {name}")
    for name in ("data.json", "prompt.txt"):
        require(evidence[name] == case["sha256"][name], "Runtime input differs from canonical input")
    policy_identity = read(directory / "identity.json")
    require(all(policy_identity.get(key) == receipt[key] for key in ("model", "mode", "max_attempts", "protocol_id")),
            "Saved policy identity differs from adjudication")
    require(policy_identity.get("inputs") == {name: case["sha256"][name] for name in ("data.json", "prompt.txt")}
            and policy_identity.get("references") == {name: sha for name, sha in case["sha256"].items() if name not in ("data.json", "prompt.txt")},
            "Saved policy input/reference identities differ")
    attempts = raw.get("attempts", [])
    index = receipt.get("terminal_attempt")
    require(isinstance(index, int) and 1 <= index <= receipt["max_attempts"], "Invalid terminal attempt")
    require([attempt.get("attempt") for attempt in attempts] == list(range(1, index + 1)),
            "Terminal summary is not an actual attempt prefix")
    require(all(not attempt.get("reused_from_previous") and attempt.get("status") != "ok" for attempt in attempts),
            "Terminal failure cannot replace a successful or padded outcome")
    require(receipt.get("actual_attempts") == len(attempts)
            and receipt.get("unexecuted_attempt_slots") == receipt["max_attempts"] - len(attempts),
            "Terminal attempt counts differ")
    for attempt in attempts:
        base = directory / "attempts" / f"attempt_{attempt['attempt']:02d}"
        score_path = base / "score_receipt.json"
        require(str(score_path.relative_to(directory)) in evidence, "A scored attempt is not pinned")
        score = read(score_path)
        require(score["result"] == attempt, "Original scored attempt differs from summary")
        require(read(base / "agent_receipt.json")["result"] == attempt.get("metrics", {}).get("generation"),
                "Scored generation receipt differs")
        for name, expected in score.get("artifact_sha256", {}).items():
            require(digest(_evidence_path(base, name)) == expected, "Original scored artifact changed")
    later = [path for path in (directory / "attempts").glob("attempt_*") if int(path.name.split("_")[-1]) > index]
    require(not later, "Later attempt artifacts exist after the terminal prefix")
    terminal = directory / "attempts" / f"attempt_{index:02d}"
    require(not (terminal / "reflection_receipt.json").exists(), "A committed reflection already exists")
    require(not (terminal / "debug.md").exists() or not (terminal / "debug.md").read_text().strip(),
            "A nonempty reflection must not be adjudicated away")
    calls = {path.parent.name: path for path in terminal.glob("calls/*/reflection_metrics.json")}
    require(bool(calls) and sorted(calls) == receipt.get("failed_reflection_call_ids"),
            "Terminal reflection call inventory differs")
    require(receipt.get("failed_reflection_calls") == len(calls), "Reflection failure count differs")
    for invocation, path in calls.items():
        metrics = read(path)
        request = read(path.parent / "request.json")
        require(str(path.relative_to(directory)) in evidence
                and str((path.parent / "request.json").relative_to(directory)) in evidence,
                "Reflection call evidence is not pinned")
        require(metrics.get("ok") is False and metrics.get("returncode") == 0
                and metrics.get("terminal_state") == "completed" and metrics.get("artifact_sha256") is None
                and metrics.get("invocation_id") == invocation
                and (metrics.get("error") or "").startswith("AGY did not complete an edit (exit 0)"),
                "Terminal closure cannot replace a successful or infrastructure-rejected call")
        require(request.get("step") == "reflection" and request.get("target") == "debug.md"
                and request.get("protocol") == receipt["protocol_id"]
                and request.get("request_sha256") == receipt.get("request_sha256"),
                "Reflection request differs from the adjudicated request")
    plain_directory = root / "runs" / receipt["model"] / "plain" / receipt["suite"] / receipt["problem"] / receipt["instance"]
    plain = read(plain_directory / "instance_summary.json")
    require(digest(plain_directory / "instance_summary.json") == receipt.get("plain_summary_sha256")
            and policy_is_complete(plain, receipt["max_attempts"], receipt["protocol_id"]),
            "Completed plain policy changed")
    shared = read(directory / "shared_first.json")
    require(shared == read(plain_directory / "shared_first.json")
            and shared.get("id") == identity({"protocol": receipt["protocol_id"], "model": receipt["model"], "instance": receipt["instance"], "attempt": 1})
            and shared.get("generation_invocation_id") == attempts[0]["metrics"]["generation"].get("invocation_id")
            and attempts[0]["metrics"]["generation"] == plain["attempts"][0]["metrics"]["generation"],
            "Shared first-attempt identity differs")


def load_policy_summary(directory: Path, summary: dict | None = None, *, campaign_root: Path | None = None,
                        approvals: dict | None = None) -> dict:
    """Return validated completion metadata while retaining the actual attempt list."""
    directory = Path(directory).resolve()
    path = directory / "instance_summary.json"
    raw = summary if summary is not None else (read(path) if path.exists() else {})
    if (directory / DELIVERY_RECEIPT).exists():
        require(not (directory / TERMINAL_RECEIPT).exists(), "Conflicting terminal receipts")
        root = Path(campaign_root).resolve() if campaign_root else campaign_root_for(directory)
        return load_delivery_summary(directory, root)
    receipt_path = directory / TERMINAL_RECEIPT
    require(isinstance(raw, dict), "Policy summary is not an object")
    if receipt_path.exists() and raw.get("completion_kind") == TERMINAL_KIND:
        raw = read(path)
    root = Path(campaign_root).resolve() if campaign_root else None
    if root is None and (receipt_path.exists() or raw.get("shared_first_attempt")):
        try:
            root = campaign_root_for(directory)
        except TerminalCompletionError:
            if receipt_path.exists():
                raise
    if root is not None and approvals is None:
        approvals = approved_receipts(root)
    if not receipt_path.exists():
        if root is not None and approvals is not None:
            require(str(receipt_path.relative_to(root)) not in approvals, "Approved terminal receipt is missing")
        require(not is_terminal_failure(raw), "Terminal summary is missing its receipt")
        return deepcopy(raw)
    require(root is not None and approvals is not None, "Terminal approval inventory is unavailable")
    relative = str(receipt_path.relative_to(root))
    receipt_hash = digest(_evidence_path(root, relative))
    require(approvals.get(relative) == receipt_hash, "Terminal receipt is not approved by the execution phase")
    receipt = read(receipt_path)
    _validate(directory, root, raw, receipt)
    effective = deepcopy(raw)
    effective.update(complete=True, completion_kind=TERMINAL_KIND, outcome="failed",
                     terminal_reason=receipt["reason"], terminal_stage="reflection",
                     terminal_attempt=receipt["terminal_attempt"], actual_attempts=receipt["actual_attempts"],
                     unexecuted_attempt_slots=receipt["unexecuted_attempt_slots"], reused_slots=0,
                     attempt_budget_exhausted=False, failed_reflection_calls=receipt["failed_reflection_calls"],
                     terminal_receipt_sha256=receipt_hash)
    return effective


def validate_delivery_call(call: Path, root: Path) -> dict:
    """Accept only a physical wrong-target write under the prospective phase rule."""
    require(call.resolve().is_relative_to((root / "runs").resolve()), "Delivery call is outside this campaign")
    relative = call.resolve().relative_to(root.resolve())
    require(len(relative.parts) == 10 and relative.parts[0] == "runs" and relative.parts[2] == "reflection"
            and relative.parts[-2] == "calls", "Wrong-target evidence is not a reflection call")
    if read(call / "reflection_workspace.json").get("version") == "existing-target-cwd-v2":
        return _validate_fresh_delivery_call(call, root, relative.parts[1])
    metadata = read(call / "execution_phase.json")
    phases = [(path, read(path)) for path in (root / "execution_phases").glob("*/phase.json")]
    matching = [(path, phase) for path, phase in phases if phase.get("phase_id") == metadata.get("phase_id")]
    require(len(matching) == 1, "Wrong-target call has no unique authorization phase")
    phase_path, phase = matching[0]
    require(identity({k: v for k, v in phase.items() if k != "phase_id"}) == phase["phase_id"], "Delivery authorization phase changed")
    require(phase.get("changes", {}).get("agy_workspace_clarification") == WORKSPACE_AMENDMENT
            and phase["changes"].get("wrong_target_reflection_failure") == DELIVERY_AMENDMENT
            and bool(phase.get("workspace_change_authorization")), "Future wrong-target failure is not authorized")
    require(digest(root / "campaign.json") == phase.get("base_manifest_sha256"), "Delivery campaign changed")
    require(digest(phase_path.parent / "agy_client.py") == phase.get("agy_client_sha256"), "Clarified adapter changed")
    require(metadata.get("role") == "reflection" and metadata.get("model") == relative.parts[1], "Delivery call role or model differs")
    request = read(call / "request.json")
    require(request.get("step") == "reflection" and request.get("target") == "debug.md"
            and request.get("protocol") == phase["base_protocol_id"], "Delivery request differs")
    require(request.get("request_sha256") == identity({k: v for k, v in request.items() if k not in ("request_sha256", "invocation_id")}), "Delivery request fingerprint changed")
    metrics = read(call / "reflection_metrics.json")
    require(metrics.get("ok") is False and metrics.get("returncode") == 0 and metrics.get("terminal_state") == "completed"
            and (metrics.get("error") or "").startswith("AGY did not complete an edit (exit 0)"), "Delivery failure is not a completed model omission")
    workspace = read(call / "reflection_workspace.json")
    require(workspace.get("version") == WORKSPACE_AMENDMENT and Path(workspace["workspace"]).is_absolute()
            and workspace.get("target") == str(Path(workspace["workspace"]) / "debug.md"), "Clarified target changed")
    note = f"For this editing call, the workspace is `{workspace['workspace']}`; edit only `{workspace['target']}`."
    before = _evidence_path(call, "reflection_prompt_before_workspace.md")
    sent = _evidence_path(call, "reflection_prompt.md")
    require(workspace.get("note") == note and digest(before) == workspace.get("before_prompt_sha256")
            and digest(sent) == workspace.get("effective_prompt_sha256")
            and sent.read_text() == note + "\n\n" + before.read_text(), "Actual prompt is not the one-sentence clarification")
    delivery = read(call / "reflection_delivery_failure.json")
    require(delivery.get("kind") == "wrong_target_file" and delivery.get("workspace_note_version") == WORKSPACE_AMENDMENT
            and delivery.get("target_name") == "debug.md" and delivery.get("required_target_empty") is True
            and delivery.get("workspace_record_sha256") == digest(call / "reflection_workspace.json"), "Wrong-target record differs")
    require(bool(delivery.get("misplaced_files")), "No wrong-target artifact was preserved")
    for artifact in delivery["misplaced_files"]:
        name = Path(artifact["relative_path"])
        require(not name.is_absolute() and ".." not in name.parts and name.name == "debug.md" and name.parent != Path("."), "Alternative target is not a nested debug.md")
        archived = _evidence_path(call, artifact["archive"])
        require(artifact["archive"] == str(Path("unexpected_outputs") / name)
                and bool(archived.read_bytes().strip()) and digest(archived) == artifact["sha256"], "Wrong-target artifact changed")
    target = call.parent.parent / "debug.md"
    require(not target.exists() or not target.read_bytes().strip(), "Required reflection was subsequently delivered")
    return {"phase_file": str(phase_path.relative_to(root)), "phase_id": phase["phase_id"], "request_sha256": request["request_sha256"]}


def _validate_fresh_delivery_call(call: Path, root: Path, model: str) -> dict:
    """The new campaign authorizes the short wording without redefining legacy phases."""
    from src.agent.prompts.policy import FEEDBACK_POLICY, WORKSPACE_POLICY, edit_instruction
    manifest = read(root / "campaign.json")
    require(manifest.get("protocol_id") == identity({k: v for k, v in manifest.items() if k != "protocol_id"}), "Delivery campaign changed")
    require(manifest.get("prompt_policies") == {"feedback": FEEDBACK_POLICY, "workspace": WORKSPACE_POLICY}
            and manifest.get("wrong_target_reflection_failure") == DELIVERY_AMENDMENT, "Fresh wrong-target failure is not authorized")
    models = {item["model"]: item for item in manifest.get("model_queue", [])}
    require(model in models, "Delivery model is outside the frozen queue")
    hashes = manifest.get("controller_sha256", {})
    require(all(name in hashes for name in ("src/agent/prompt_policy.py", "src/agent/policy_completion.py", "src/agent/paired_runner.py")), "Frozen delivery controller is incomplete")
    for name, expected in hashes.items():
        require(digest(_evidence_path(root / "controller", name)) == expected, "Frozen delivery controller changed")
    request = read(call / "request.json")
    require(request.get("step") == "reflection" and request.get("target") == "debug.md"
            and request.get("protocol") == manifest["protocol_id"] and request.get("prompt_policies") == manifest["prompt_policies"], "Delivery request differs")
    require(request.get("request_sha256") == identity({k: v for k, v in request.items() if k not in ("request_sha256", "invocation_id")}), "Delivery request fingerprint changed")
    metrics = read(call / "reflection_metrics.json")
    require(metrics.get("ok") is False and metrics.get("returncode") == 0 and metrics.get("terminal_state") == "completed", "Delivery failure is not a completed model omission")
    invocation = read(call / "reflection_invocation.json")
    require(invocation.get("backend") == models[model]["backend"], "Delivery backend differs")
    workspace = read(call / "reflection_workspace.json")
    require(workspace.get("version") == WORKSPACE_POLICY and Path(workspace["workspace"]).is_absolute()
            and workspace.get("target") == str(Path(workspace["workspace"]) / "debug.md"), "Clarified target changed")
    before = _evidence_path(call, "reflection_prompt_before_workspace.md")
    sent = _evidence_path(call, "reflection_prompt.md")
    old = before.read_text()
    if workspace.get("before_policy") == WORKSPACE_AMENDMENT:
        legacy_note = f"For this editing call, the workspace is `{workspace['workspace']}`; edit only `{workspace['target']}`.\n\n"
        require(old.startswith(legacy_note), "Legacy comparison note differs")
        old = old[len(legacy_note):]
    else:
        require(workspace.get("before_policy") == "edit-only-basename-v0", "Unknown comparison policy")
    note = edit_instruction("debug.md")
    require(old.startswith("Edit only debug.md.") and workspace.get("note") == note
            and sent.read_text() == note + old[len("Edit only debug.md."):]
            and digest(before) == workspace.get("before_prompt_sha256") and digest(sent) == workspace.get("effective_prompt_sha256"), "Actual prompt differs from the short instruction")
    delivery = read(call / "reflection_delivery_failure.json")
    require(delivery.get("kind") == "wrong_target_file" and delivery.get("workspace_note_version") == WORKSPACE_POLICY
            and delivery.get("target_name") == "debug.md" and delivery.get("required_target_empty") is True
            and delivery.get("workspace_record_sha256") == digest(call / "reflection_workspace.json"), "Wrong-target record differs")
    require(bool(delivery.get("misplaced_files")), "No wrong-target artifact was preserved")
    for artifact in delivery["misplaced_files"]:
        name = Path(artifact["relative_path"])
        require(not name.is_absolute() and ".." not in name.parts and name.name == "debug.md" and name.parent != Path("."), "Alternative target is not a nested debug.md")
        archived = _evidence_path(call, artifact["archive"])
        require(artifact["archive"] == str(Path("unexpected_outputs") / name)
                and bool(archived.read_bytes().strip()) and digest(archived) == artifact["sha256"], "Wrong-target artifact changed")
    target = call.parent.parent / "debug.md"
    require(not target.exists() or not target.read_bytes().strip(), "Required reflection was subsequently delivered")
    return {"campaign_file": "campaign.json", "protocol_id": manifest["protocol_id"], "request_sha256": request["request_sha256"]}


def build_delivery_receipt(directory: Path, attempt: Path, root: Path, *, recorded_at: str) -> dict:
    """Close one future failed delivery while preserving the raw summary and scores."""
    raw = read(directory / "instance_summary.json")
    stage = read(attempt / "reflection_receipt.json")
    require(stage.get("result", {}).get("file_delivery_failure") is True, "No committed wrong-target failure")
    call = attempt / "calls" / stage["result"]["invocation_id"]
    authorization = validate_delivery_call(call, root)
    index = int(attempt.name.split("_")[-1])
    scores = [read(directory / "attempts" / f"attempt_{n:02d}" / "score_receipt.json")["result"] for n in range(1, index + 1)]
    receipt = {"schema_version": 1, "kind": DELIVERY_KIND, "recorded_at": recorded_at,
               "authorization": authorization, "outcome": "failed", "mode": "reflection", "stage": "reflection",
               "reason": "required_reflection_written_to_wrong_path", "trigger_call": str(call.relative_to(directory)),
               **{k: raw[k] for k in ("model", "suite", "problem", "instance", "protocol_id", "max_attempts")},
               "terminal_attempt": index, "actual_attempts": index, "unexecuted_attempt_slots": raw["max_attempts"] - index,
               "attempts": scores, "evidence_sha256": {str(p.relative_to(directory)): digest(p) for p in directory.rglob("*") if p.is_file()}}
    plain = root / "runs" / raw["model"] / "plain" / raw["suite"] / raw["problem"] / raw["instance"]
    receipt["plain_summary_sha256"] = digest(plain / "instance_summary.json")
    receipt["receipt_id"] = identity(receipt)
    validate_delivery_receipt(directory, root, receipt)
    return receipt


def validate_delivery_receipt(directory: Path, root: Path, receipt: dict):
    require(receipt.get("receipt_id") == identity({k: v for k, v in receipt.items() if k != "receipt_id"}), "Delivery receipt changed")
    require(receipt.get("schema_version") == 1 and receipt.get("kind") == DELIVERY_KIND
            and receipt.get("outcome") == "failed" and receipt.get("mode") == "reflection", "Invalid delivery completion")
    raw = read(directory / "instance_summary.json")
    require(raw.get("reflection") is True and not raw.get("complete"), "Delivery closure must preserve unfinished raw policy")
    expected = root / "runs" / receipt["model"] / "reflection" / receipt["suite"] / receipt["problem"] / receipt["instance"]
    require(directory.resolve() == expected.resolve(), "Delivery policy identity differs")
    require(all(raw.get(k) == receipt.get(k) for k in ("model", "suite", "problem", "instance", "protocol_id", "max_attempts")), "Delivery summary identity differs")
    manifest = read(root / "campaign.json")
    case = next((c for c in manifest["instances"] if all(c[k] == receipt[k] for k in ("suite", "problem", "instance"))), None)
    require(case is not None and receipt["protocol_id"] == manifest["protocol_id"] and receipt["max_attempts"] == manifest["max_attempts"], "Delivery case is outside the fixed protocol")
    for name, expected_hash in case["sha256"].items():
        require(digest(root / "dataset" / case["suite"] / case["problem"] / case["instance"] / name) == expected_hash, "Canonical delivery input changed")
    for name in ("data.json", "prompt.txt"):
        require(digest(directory / name) == case["sha256"][name], "Delivery runtime input changed")
    plain = root / "runs" / raw["model"] / "plain" / raw["suite"] / raw["problem"] / raw["instance"]
    require(digest(plain / "instance_summary.json") == receipt["plain_summary_sha256"]
            and policy_is_complete(read(plain / "instance_summary.json"), receipt["max_attempts"], receipt["protocol_id"]), "Plain policy is not intact and complete")
    evidence = receipt.get("evidence_sha256", {})
    require(all(name in evidence for name in ("instance_summary.json", "identity.json", "shared_first.json", "data.json", "prompt.txt")), "Delivery identity evidence is incomplete")
    for name, expected_hash in evidence.items():
        require(digest(_evidence_path(directory, name)) == expected_hash, "Delivery evidence changed: " + name)
    call = directory / receipt["trigger_call"]
    require(call.resolve().is_relative_to(directory.resolve()) and validate_delivery_call(call, root) == receipt["authorization"], "Delivery trigger authorization differs")
    stage = read(call.parent.parent / "reflection_receipt.json")
    require(stage["result"] == read(call / "reflection_metrics.json") and stage["result"].get("file_delivery_failure") is True
            and stage.get("request_sha256") == receipt["authorization"]["request_sha256"], "Failed reflection commit changed")
    index = receipt["terminal_attempt"]
    require(isinstance(index, int) and 1 <= index <= receipt["max_attempts"] and receipt["actual_attempts"] == index
            and receipt["unexecuted_attempt_slots"] == receipt["max_attempts"] - index, "Delivery attempt accounting differs")
    require([a.get("attempt") for a in receipt["attempts"]] == list(range(1, index + 1)), "Delivery score prefix is incomplete")
    for number, result in enumerate(receipt["attempts"], 1):
        attempt = directory / "attempts" / f"attempt_{number:02d}"
        score = read(attempt / "score_receipt.json")
        require(score["result"] == result and result["status"] not in ("ok", "skipped_previous_solved"), "Delivery closure changed a score or passed candidate")
        for name, expected_hash in score.get("artifact_sha256", {}).items():
            require(digest(_evidence_path(attempt, name)) == expected_hash, "Delivery candidate changed")
    require(not any(int(p.name.split("_")[-1]) > index for p in (directory / "attempts").glob("attempt_*")), "Later attempts exist after delivery closure")


def load_delivery_summary(directory: Path, root: Path) -> dict:
    receipt = read(directory / DELIVERY_RECEIPT)
    validate_delivery_receipt(directory, root, receipt)
    effective = deepcopy(read(directory / "instance_summary.json"))
    effective.update(complete=True, completion_kind=DELIVERY_KIND, outcome="failed", terminal_reason=receipt["reason"],
                     terminal_stage="reflection", terminal_attempt=receipt["terminal_attempt"], attempts=receipt["attempts"],
                     actual_attempts=receipt["actual_attempts"], unexecuted_attempt_slots=receipt["unexecuted_attempt_slots"],
                     reused_slots=0, attempt_budget_exhausted=receipt["unexecuted_attempt_slots"] == 0,
                     terminal_receipt_sha256=digest(directory / DELIVERY_RECEIPT))
    for attempt in effective["attempts"]:
        reflection = directory / "attempts" / f"attempt_{attempt['attempt']:02d}" / "reflection_receipt.json"
        if reflection.exists():
            attempt.setdefault("metrics", {})["reflection"] = read(reflection)["result"]
    return effective


def build_terminal_receipt(directory: Path, root: Path, *, user_decision: str, recorded_at: str,
                           source_phase_id: str) -> dict:
    """Prepare one explicit adjudication; does not write or authorize the receipt."""
    directory, root = directory.resolve(), root.resolve()
    require(not (directory / TERMINAL_RECEIPT).exists(), "Terminal receipt already exists")
    raw, manifest = read(directory / "instance_summary.json"), read(root / "campaign.json")
    attempts = raw.get("attempts", [])
    require(bool(attempts), "No scored attempt exists to close")
    case = next((case for case in manifest["instances"] if case["instance"] == raw["instance"]), None)
    require(case is not None, "Unknown adjudicated instance")
    index = attempts[-1]["attempt"]
    terminal = directory / "attempts" / f"attempt_{index:02d}"
    calls = sorted(terminal.glob("calls/*/reflection_metrics.json"))
    require(bool(calls), "No failed physical reflection calls exist")
    request = read(calls[0].parent / "request.json")
    plain = root / "runs" / raw["model"] / "plain" / raw["suite"] / raw["problem"] / raw["instance"]
    receipt = {"schema_version": 1, "kind": TERMINAL_KIND, "recorded_at": recorded_at,
               "authorization": {"scope": "one_saved_policy", "user_decision": user_decision},
               "source_execution_phase_id": source_phase_id, "outcome": "failed", "stage": "reflection",
               "reason": "required_reflection_artifact_not_produced", "mode": "reflection",
               **{key: raw[key] for key in ("protocol_id", "model", "suite", "problem", "instance", "max_attempts")},
               "terminal_attempt": index, "actual_attempts": len(attempts),
               "unexecuted_attempt_slots": raw["max_attempts"] - len(attempts),
               "failed_reflection_calls": len(calls), "failed_reflection_call_ids": sorted(path.parent.name for path in calls),
               "request_sha256": request["request_sha256"], "canonical_input_sha256": case["sha256"],
               "plain_summary_sha256": digest(plain / "instance_summary.json"),
               "evidence_sha256": {str(path.relative_to(directory)): digest(path) for path in sorted(directory.rglob("*")) if path.is_file()},
               "note": "Failed reflection calls are retries of a saved candidate. Unexecuted budget slots are not generated attempts."}
    receipt["receipt_id"] = identity(receipt)
    _validate(directory, root, raw, receipt)
    return receipt
