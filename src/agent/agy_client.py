"""Native, account-backed AGY file editing in a fresh temporary workspace."""
from __future__ import annotations

import hashlib
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from datetime import datetime, timezone

from src.agent.codex_client import CodexClient, CodexResult
from src.agent.resource_monitor import stop_owned_tree
from src.agent.prompt_policy import WORKSPACE_POLICY, record_workspace_prompt, record_wrong_target


WORKSPACE_NOTE_VERSION = WORKSPACE_POLICY


class AgyClient:
    prompt_journal = None

    def __init__(self, model: str, *, timeout: int = 900, role: str = "generation", executable: str = "agy"):
        self.model, self.timeout, self.role, self.executable = model, timeout, role, executable

    def protocol(self):
        return {"backend": "agy-cli", "model": self.model, "timeout": self.timeout, "role": self.role, "effort": "catalog-slug", "workspace_policy": WORKSPACE_POLICY, "isolation": "fresh workspace/conversation; native user settings retained; slash expansion disabled"}

    def edit_file(self, target: Path, prompt: str, *, context=None, log_dir=None, step="agent"):
        target = target.resolve()
        log_dir = (log_dir or target.parent).resolve()
        log_dir.mkdir(parents=True, exist_ok=True)
        executable = shutil.which(self.executable)
        if executable is None:
            result = CodexResult(False, None, 0, {}, "AGY CLI not found")
            CodexClient._save_result(log_dir, step, result)
            return result
        sections = [f"Edit only {target.name}. All task context is supplied below.", "Do not inspect other directories, use external tools or services, or search for reference implementations. Do not execute the generated solver. Do not use memory, skills, subagents or prior conversations. Keep prose paragraphs on one physical line.", prompt]
        for path in context or []:
            if path.exists():
                sections.append(f"--- BEGIN CONTEXT: {path.parent.name}/{path.name} ---\n{path.read_text()}\n--- END CONTEXT ---")
        effective = "\n\n".join(sections)
        started = time.perf_counter()
        error, returncode, terminal, usage, artifact_hash = None, None, None, {}, None
        with tempfile.TemporaryDirectory(prefix="decompbench-agy-") as temporary:
            workspace = Path(temporary).resolve()
            staged = workspace / target.name
            staged.write_text(target.read_text() if target.exists() else "")
            effective = record_workspace_prompt(workspace, target.name, effective, log_dir, step, old_absolute_note=True)
            command = [executable, "--input-format", "stream-json", "--output-format", "stream-json", "--model", self.model, "--new-project", "--mode", "accept-edits", "--sandbox", "--disable-slash-commands", "--print-timeout", f"{self.timeout}s", "--log-file", os.devnull]
            (log_dir / f"{step}_invocation.json").write_text(json.dumps(self.protocol() | {"target": target.name, "fresh_conversation": True, "resume": False, "ignore_user_config": False, "sandbox": True}, indent=2))
            with (log_dir / f"{step}_events.jsonl").open("w") as events, (log_dir / f"{step}_stderr.log").open("w") as stderr:
                process = subprocess.Popen(command, cwd=workspace, stdin=subprocess.PIPE, stdout=events, stderr=stderr, text=True, start_new_session=True, env=os.environ | {"AGY_CLI_DISABLE_AUTO_UPDATE": "true"})
                try:
                    process.communicate(json.dumps({"event": "user", "message": {"content": effective}}) + "\n", timeout=self.timeout)
                except subprocess.TimeoutExpired:
                    stop_owned_tree(process)
                    error = f"AGY edit timed out after {self.timeout} seconds"
                returncode = process.returncode
            for line in (log_dir / f"{step}_events.jsonl").read_text().splitlines():
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event.get("event") == "result":
                    payload = event.get("result", {})
                    terminal = "completed" if payload.get("status") == "SUCCESS" else "failed"
                    usage = {k: v for k, v in payload.get("usage", {}).items() if isinstance(v, int)}
                    if "cache_read_tokens" in usage:
                        usage["cached_input_tokens"] = usage["cache_read_tokens"]
                    error = payload.get("error") or error
                    (log_dir / f"{step}_response.md").write_text(payload.get("response", ""))
            ok = returncode == 0 and terminal == "completed" and error is None and staged.is_file() and bool(staged.read_text().strip())
            if returncode == 0 and terminal == "completed" and error is None and not ok:
                record_wrong_target(workspace, target.name, log_dir, step)
            if ok:
                shutil.copyfile(staged, target)
                artifact_hash = hashlib.sha256(target.read_bytes()).hexdigest()
            elif error is None:
                error = f"AGY did not complete an edit (exit {returncode}); see {step}_stderr.log"
        result = CodexResult(ok, returncode, time.perf_counter() - started, usage, error, [], terminal, artifact_hash)
        CodexClient._save_result(log_dir, step, result)
        if self.prompt_journal:
            journal = Path(self.prompt_journal)
            with journal.open("a") as stream:
                fcntl.flock(stream, fcntl.LOCK_EX)
                stamp = datetime.now(timezone.utc).isoformat()
                stream.write(f"\n- {stamp} | {self.model} | {step} | `{log_dir}` | ok={result.ok} | [同次调用的原版对照]({log_dir / (step + '_prompt_before_workspace.md')}) → [实际发送的修改版]({log_dir / (step + '_prompt.md')}) | [路径及文本哈希]({log_dir / (step + '_workspace.json')})\n")
        return result
