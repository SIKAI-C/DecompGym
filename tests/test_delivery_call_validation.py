"""A fresh campaign's wrong-target reflection call validates under both workspace records: the pasted-context record
(existing-target-cwd-v2) and the mounted-files record (existing-target-cwd-mounted-files-v1), with the same checks."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.agent.prompts.mounted import CONTEXT_DELIVERY_MOUNTED, MOUNTED_WORKSPACE_POLICY, workspace_note
from src.agent.prompts.policy import FEEDBACK_POLICY, WORKSPACE_POLICY, record_workspace_prompt
from src.experiments.completion import DELIVERY_AMENDMENT, TerminalCompletionError, digest, identity, validate_delivery_call

MODEL = "tested-model"


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) if not isinstance(value, str) else value)


def fresh_call(tmp_path, *, mounted, campaign_mounted=None, note=None):
    root = tmp_path / "campaign"
    hashes = {}
    for name in ("src/agent/prompt_policy.py", "src/agent/policy_completion.py", "src/agent/paired_runner.py"):
        write(root / "controller" / name, "# frozen controller file\n")
        hashes[name] = digest(root / "controller" / name)
    campaign_mounted = mounted if campaign_mounted is None else campaign_mounted
    manifest = {"prompt_policies": {"feedback": FEEDBACK_POLICY, "workspace": WORKSPACE_POLICY}, "wrong_target_reflection_failure": DELIVERY_AMENDMENT,
                "model_queue": [{"model": MODEL, "backend": "codex-cli", "effort": "medium"}], "controller_sha256": hashes,
                "prompt_protocol": {"context_delivery": CONTEXT_DELIVERY_MOUNTED} if campaign_mounted else {}}
    manifest["protocol_id"] = identity(manifest)
    write(root / "campaign.json", manifest)
    call = root / "runs" / MODEL / "reflection/bd/ufl/ufl_small_seed000/attempts/attempt_01/calls/c1"
    request = {"step": "reflection", "target": "debug.md", "protocol": manifest["protocol_id"], "prompt_policies": manifest["prompt_policies"], "prompt": "Reflect."}
    request |= {"request_sha256": identity(request), "invocation_id": "c1"}
    write(call / "request.json", request)
    write(call / "reflection_metrics.json", {"ok": False, "returncode": 0, "terminal_state": "completed"})
    write(call / "reflection_invocation.json", {"backend": "codex-cli"})
    if mounted:
        composed = "Edit only debug.md. The only file you may create or modify is /work/debug.md; every other path in this workspace is read-only context.\n\n# Part 2: Task\n\nReflect."
        record_workspace_prompt(Path("/work"), "debug.md", composed, call, "reflection", note=note or workspace_note("debug.md"), version=MOUNTED_WORKSPACE_POLICY)
    else:
        record_workspace_prompt(Path("/work"), "debug.md", "Edit only debug.md. All task context is supplied below.\n\nReflect.", call, "reflection", note=note)
    write(call / "unexpected_outputs/nested/debug.md", "misplaced reflection")
    write(call / "reflection_delivery_failure.json", {"kind": "wrong_target_file", "workspace_note_version": WORKSPACE_POLICY, "target_name": "debug.md", "required_target_empty": True,
                                                      "workspace_record_sha256": digest(call / "reflection_workspace.json"),
                                                      "misplaced_files": [{"relative_path": "nested/debug.md", "archive": "unexpected_outputs/nested/debug.md",
                                                                           "sha256": digest(call / "unexpected_outputs/nested/debug.md")}]})
    return root, call, manifest, request


@pytest.mark.parametrize("mounted", [False, True])
def test_both_workspace_records_pass_the_same_fresh_checks(tmp_path, mounted):
    root, call, manifest, request = fresh_call(tmp_path, mounted=mounted)
    assert json.loads((call / "reflection_workspace.json").read_text())["version"] == (MOUNTED_WORKSPACE_POLICY if mounted else WORKSPACE_POLICY)
    assert validate_delivery_call(call, root) == {"campaign_file": "campaign.json", "protocol_id": manifest["protocol_id"], "request_sha256": request["request_sha256"]}
    (call.parent.parent / "debug.md").write_text("delivered later")
    with pytest.raises(TerminalCompletionError, match="subsequently delivered"):
        validate_delivery_call(call, root)


def test_a_mounted_record_needs_a_mounted_campaign_and_its_part1_note(tmp_path):
    root, call, _, _ = fresh_call(tmp_path / "pasted", mounted=True, campaign_mounted=False)
    with pytest.raises(TerminalCompletionError, match="outside a mounted-files campaign"):
        validate_delivery_call(call, root)
    root, call, _, _ = fresh_call(tmp_path / "note", mounted=True, note="Only edit the existing debug.md in the current working directory.")
    with pytest.raises(TerminalCompletionError, match="short instruction"):
        validate_delivery_call(call, root)
