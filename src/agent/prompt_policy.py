"""Versioned agent-facing feedback and existing-file editing instructions."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil


FEEDBACK_POLICY = "qualitative-objective-feedback-v1"
WORKSPACE_POLICY = "existing-target-cwd-v2"
OBJECTIVE_RETRY = "The current solution is feasible, but its objective value does not meet the required target. Please try again."


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


def record_wrong_target(workspace, filename, log_dir, step):
    """Retain evidence of a completed wrong-path write; never promote that file."""
    misplaced = []
    staged = workspace / filename
    for path in sorted(workspace.rglob(filename)):
        if path == staged or path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(workspace) or not path.read_bytes().strip():
            continue
        relative = path.relative_to(workspace)
        archive = log_dir / "unexpected_outputs" / relative
        archive.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, archive)
        misplaced.append({"relative_path": str(relative), "archive": str(archive.relative_to(log_dir)), "sha256": hashlib.sha256(archive.read_bytes()).hexdigest()})
    if misplaced:
        record = {"kind": "wrong_target_file", "workspace_note_version": WORKSPACE_POLICY, "target_name": filename,
                  "required_target_empty": True, "workspace_record_sha256": hashlib.sha256((log_dir / f"{step}_workspace.json").read_bytes()).hexdigest(), "misplaced_files": misplaced}
        (log_dir / f"{step}_delivery_failure.json").write_text(json.dumps(record, indent=2) + "\n")
