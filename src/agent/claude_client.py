"""Claude Code file edits with pinned models and a fresh restricted workspace."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

from src.agent.codex_client import CodexClient, CodexResult
from src.agent.prompt_policy import WORKSPACE_POLICY, record_workspace_prompt, record_wrong_target
from src.agent.resource_monitor import stop_owned_tree


class ClaudeClient:
    def __init__(self, model: str, *, effort: str = "high", timeout: int = 900, role: str = "generation", executable: str = "claude"):
        self.model, self.effort, self.timeout, self.role, self.executable = model, effort, timeout, role, executable

    def protocol(self):
        return {"backend": "claude-code-cli", "model": self.model, "effort": self.effort, "timeout": self.timeout, "role": self.role,
                "workspace_policy": WORKSPACE_POLICY, "isolation": "fresh restricted workspace; safe mode; read/edit/write tools only; no session persistence"}

    def edit_file(self, target: Path, prompt: str, *, context=None, log_dir=None, step="agent"):
        target = target.resolve()
        log_dir = (log_dir or target.parent).resolve()
        log_dir.mkdir(parents=True, exist_ok=True)
        executable = shutil.which(self.executable)
        if executable is None:
            result = CodexResult(False, None, 0, {}, "Claude Code CLI not found")
            CodexClient._save_result(log_dir, step, result)
            return result
        sections = [f"Edit only {target.name}. All task context is supplied below.",
                    "Do not inspect other directories, use external tools or services, or search for reference implementations.",
                    "Do not execute the generated solver. The harness executes it later in its documented runtime directory.",
                    "Keep prose paragraphs and list items on one physical line; rely on editor soft wrapping.", prompt]
        for path in context or []:
            if path.exists():
                label = f"{path.parent.name}/{path.name}"
                sections.append(f"--- BEGIN CONTEXT: {label} ---\n{path.read_text()}\n--- END CONTEXT: {label} ---")
        started = time.perf_counter()
        error, returncode, terminal, usage, observed = None, None, None, {}, set()
        with tempfile.TemporaryDirectory(prefix="decompbench-claude-") as temporary:
            workspace = Path(temporary).resolve()
            staged = workspace / target.name
            staged.write_text(target.read_text() if target.exists() else "")
            effective = record_workspace_prompt(workspace, target.name, "\n\n".join(sections), log_dir, step)
            command = [executable, "--print", "--model", self.model, "--effort", self.effort,
                       "--input-format", "text", "--output-format", "stream-json", "--verbose",
                       "--no-session-persistence", "--restricted", "--safe-mode", "--no-chrome",
                       "--tools", "Read,Edit,Write", "--allowedTools", "Read,Edit,Write",
                       "--permission-mode", "acceptEdits", "--permission-prompts", "none",
                       "--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}', "--disable-slash-commands"]
            invocation = self.protocol() | {"target": target.name, "fresh_conversation": True, "resume": False,
                                            "tools": ["Read", "Edit", "Write"], "permission_mode": "acceptEdits", "fallback_model": None}
            invocation_path = log_dir / f"{step}_invocation.json"
            invocation_path.write_text(json.dumps(invocation, indent=2) + "\n")
            with (log_dir / f"{step}_events.jsonl").open("w") as events, (log_dir / f"{step}_stderr.log").open("w") as stderr:
                process = subprocess.Popen(command, cwd=workspace, stdin=subprocess.PIPE, stdout=events, stderr=stderr,
                                           text=True, start_new_session=True, env=os.environ | {"DISABLE_AUTOUPDATER": "1"})
                try:
                    process.communicate(effective, timeout=self.timeout)
                except subprocess.TimeoutExpired:
                    stop_owned_tree(process)
                    error = f"Claude Code edit timed out after {self.timeout} seconds"
                returncode = process.returncode
            for line in (log_dir / f"{step}_events.jsonl").read_text().splitlines():
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event.get("type") == "assistant" and event.get("message", {}).get("model"):
                    observed.add(event["message"]["model"])
                if event.get("type") == "result":
                    terminal = "completed" if event.get("subtype") == "success" and not event.get("is_error") else "failed"
                    usage = {k: v for k, v in event.get("usage", {}).items() if type(v) is int}
                    if "cache_read_input_tokens" in usage:
                        usage["cached_input_tokens"] = usage["cache_read_input_tokens"]
                    observed.update(event.get("modelUsage", {}))
                    if terminal != "completed":
                        error = str(event.get("errors") or event.get("result") or event.get("subtype"))
                    (log_dir / f"{step}_response.md").write_text(str(event.get("result") or ""))
            invocation["observed_models"] = sorted(observed)
            invocation_path.write_text(json.dumps(invocation, indent=2) + "\n")
            if terminal == "completed" and (not observed or any(model != self.model and not model.startswith(self.model + "-") for model in observed)):
                error = f"Claude Code model identity unavailable or mismatched: requested {self.model}, observed {sorted(observed)}"
                terminal = "infrastructure_failed"
            ok = returncode == 0 and terminal == "completed" and error is None and staged.is_file() and bool(staged.read_text().strip())
            if returncode == 0 and terminal == "completed" and error is None and not ok:
                record_wrong_target(workspace, target.name, log_dir, step)
            if ok:
                shutil.copyfile(staged, target)
            elif error is None:
                error = f"Claude Code did not complete an edit (exit {returncode}); see {step}_stderr.log"
        result = CodexResult(ok, returncode, time.perf_counter() - started, usage, error, [], terminal,
                             hashlib.sha256(target.read_bytes()).hexdigest() if ok else None)
        CodexClient._save_result(log_dir, step, result)
        return result
