"""Versioned agent-facing feedback and existing-file editing instructions."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


FEEDBACK_POLICY = "qualitative-objective-feedback-v1"
WORKSPACE_POLICY = "existing-target-cwd-v2"
REFLECTION_POLICY = "consume-reflection-opportunity-and-continue-v1"
OBJECTIVE_RETRY = "The current solution is feasible, but its objective value does not meet the required target. Please try again."


def reflection_delivery_feedback(attempt: int) -> str:
    return f"[framework reflection delivery status] The reflection call after attempt {attempt} completed, but the required debug.md is empty; no usable reflection was delivered for that attempt. This is framework feedback, not a model-generated reflection."


def edit_instruction(filename: str) -> str:
    if Path(filename).name != filename:
        raise ValueError("The editing instruction requires a basename")
    return f"Only edit the existing {filename} in the current working directory."


def record_workspace_prompt(workspace, filename, old_prompt, log_dir, step, *, old_absolute_note=False):
    """Archive a before/after comparison; only the returned prompt is sent."""
    legacy_prefix = f"Edit only {filename}."
    if not old_prompt.startswith(legacy_prefix):
        raise ValueError("Unexpected editing prompt prefix")
    note = edit_instruction(filename)
    effective = note + old_prompt[len(legacy_prefix):]
    before = old_prompt
    if old_absolute_note:
        before = f"For this editing call, the workspace is `{workspace}`; edit only `{workspace / filename}`.\n\n" + before
    (log_dir / f"{step}_prompt_before_workspace.md").write_text(before)
    (log_dir / f"{step}_prompt.md").write_text(effective)
    record = {"version": WORKSPACE_POLICY, "workspace": str(workspace), "target": str(workspace / filename), "note": note,
              "before_policy": "explicit-workspace-target-v1" if old_absolute_note else "edit-only-basename-v0",
              "before_is_comparison_only": True, "before_prompt_sha256": hashlib.sha256(before.encode()).hexdigest(),
              "effective_prompt_sha256": hashlib.sha256(effective.encode()).hexdigest()}
    (log_dir / f"{step}_workspace.json").write_text(json.dumps(record, indent=2) + "\n")
    return effective
