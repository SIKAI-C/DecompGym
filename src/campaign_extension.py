"""Narrow validation and admission guards for the authorized Claude extension."""
from __future__ import annotations

import subprocess
from pathlib import Path

from src.agent.paired_runner import ProtocolError, identity, read
from src.campaign_plan import CLAUDE_EXTENSION_MODELS, CLAUDE_EXTENSION_PARENT, CLAUDE_EXTENSION_POLICY


def extension_parent(manifest):
    plan = (manifest or {}).get("experiment_plan", {})
    return plan.get("extension_parent") if plan.get("queue_policy") == CLAUDE_EXTENSION_POLICY else None


def validate_lineage(root, manifest):
    parent = extension_parent(manifest)
    if parent is None:
        return
    from src.native_campaign import validate
    parent_root = Path(parent["root"]).resolve()
    if parent_root == Path(root).resolve() or parent.get("protocol_id") != CLAUDE_EXTENSION_PARENT:
        raise ProtocolError("Invalid extension parent")
    original = validate(parent_root)
    if original["protocol_id"] != parent["protocol_id"]:
        raise ProtocolError("Extension parent identity changed")
    queue = manifest.get("model_queue", [])
    if (tuple(item.get("model") for item in queue) != CLAUDE_EXTENSION_MODELS
            or any(item.get("backend") != "claude-code-cli" or item.get("effort") != "medium" for item in queue)):
        raise ProtocolError("Extension model order or effort changed")
    scientific_fields = ("prompt_protocol", "prompt_policies", "algorithm_judge_prompt_sha256", "execution_environment", "workers", "max_attempts", "solver_timeout", "solver_threads", "agent_timeout", "solver_cooldown_seconds", "judge_model", "judge_effort", "gap_threshold", "shared_first_attempt", "stop_after_success", "reflection_policy", "resource_guard")
    if any(manifest.get(key) != original.get(key) for key in scientific_fields):
        raise ProtocolError("Extension scientific settings differ from the parent")
    def inventory(value):
        return [(case["suite"], case["problem"], case["instance"], case["sha256"]) for case in value["instances"]]
    if inventory(manifest) != inventory(original):
        raise ProtocolError("Extension dataset or rendered problem prompts differ from the parent")


def process_identity(pid):
    """Read identity without reading a process environment or its secrets."""
    result = subprocess.run(["/bin/ps", "-p", str(int(pid)), "-o", "pid=,ppid=,lstart=,stat=,command="], capture_output=True, text=True, timeout=5)
    if result.returncode or not result.stdout.strip():
        return None
    parts = result.stdout.strip().split(None, 8)
    if len(parts) != 9:
        raise RuntimeError("Cannot parse process identity")
    return {"pid": int(parts[0]), "ppid": int(parts[1]), "started": " ".join(parts[2:7]), "state": parts[7], "command": parts[8]}


def same_process(expected, actual):
    return bool(actual and all(expected.get(key) == actual.get(key) for key in ("pid", "started", "command")))


def parent_paused(root):
    manifest = read(Path(root) / "campaign.json") or {}
    parent = extension_parent(manifest)
    if parent is None:
        return False
    parent_root = Path(parent["root"])
    pause = read(parent_root / "pause_request.json") or {}
    return (parent_root / "STOP").exists() or bool(pause.get("resume_requires_explicit_user_request"))


def require_parked_parent(root, manifest):
    parent = extension_parent(manifest)
    if parent is None:
        return
    record = read(Path(parent["coordinator_state"])) or {}
    if (record.get("parent_protocol_id") != parent["protocol_id"]
            or record.get("extension_protocol_id") != manifest["protocol_id"]
            or record.get("extension_root") != str(Path(root).resolve())
            or record.get("phase") not in {"parked", "launch_intent", "extension_running"}):
        raise ProtocolError("Extension lacks a matching coordinator hold")
    expected = record.get("parent_watchdog", {})
    actual = process_identity(expected.get("pid", 0))
    if not same_process(expected, actual) or "T" not in actual["state"]:
        raise ProtocolError("Credential-bearing parent watchdog is not safely parked")


def watcher_lock(root, manifest):
    if extension_parent(manifest) is None:
        return "fixed81-native-watchdog", True
    require_parked_parent(root, manifest)
    return "fixed81-claude-extension-" + manifest["protocol_id"][:24], False


def validate_group(config):
    """A group is an additive view of two real immutable protocols, never a manifest rewrite."""
    members = config.get("members", [])
    if config.get("schema_version") != 1 or len(members) != 2:
        raise ValueError("Expected one original cohort and its approved Claude extension")
    manifests = []
    for member in members:
        manifest = read(Path(member["root"]) / "campaign.json")
        if (not manifest or manifest.get("protocol_id") != member.get("protocol_id")
                or identity({k: v for k, v in manifest.items() if k != "protocol_id"}) != member["protocol_id"]):
            raise ValueError("Group member manifest identity differs")
        manifests.append(manifest)
    parent = extension_parent(manifests[1])
    if (manifests[0]["protocol_id"] != CLAUDE_EXTENSION_PARENT or not parent
            or Path(parent["root"]).resolve() != Path(members[0]["root"]).resolve()
            or parent["protocol_id"] != manifests[0]["protocol_id"]):
        raise ValueError("Group extension lineage differs")
    if config.get("active_root") not in [member["root"] for member in members]:
        raise ValueError("Group active root is not a member")
    return manifests
