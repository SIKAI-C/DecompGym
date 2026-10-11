"""Fresh, prompt-scoped file editing through the locally installed Codex CLI."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import tempfile
import time
from typing import Any

from src.agent.prompt_policy import WORKSPACE_POLICY, record_workspace_prompt, record_wrong_target


@dataclass
class CodexResult:
    ok: bool
    returncode: int | None
    elapsed_seconds: float
    usage: dict[str, int]
    error: str | None = None
    warnings: list[str] = field(default_factory=list)
    terminal_state: str | None = None
    artifact_sha256: str | None = None


class CodexClient:
    """Start one fresh CLI session per edit and copy back only the requested file."""

    def __init__(self, model: str | None = None, *, executable: str = "codex", timeout: int = 900, effort: str | None = None, role: str = "generation") -> None:
        self.model = model
        self.executable = executable
        self.timeout = timeout
        self.effort, self.role = effort, role

    def protocol(self) -> dict:
        return {"backend": "codex-cli", "model": self.model, "effort": self.effort, "role": self.role, "timeout": self.timeout, "workspace_policy": WORKSPACE_POLICY}

    def edit_file(
        self,
        target: Path,
        prompt: str,
        *,
        context: list[Path] | None = None,
        log_dir: Path | None = None,
        step: str = "codex",
    ) -> CodexResult:
        target = target.resolve()
        log_dir = (log_dir or target.parent).resolve()
        log_dir.mkdir(parents=True, exist_ok=True)
        executable = shutil.which(self.executable)
        if executable is None:
            result = CodexResult(False, None, 0.0, {}, f"Codex CLI not found: {self.executable}")
            self._save_result(log_dir, step, result)
            return result
        sections = [
            f"Edit only {target.name}. All task context is supplied below.",
            "Do not inspect other directories, use external tools or services, or search for reference implementations.",
            "Do not execute the generated solver. The harness executes it later in its documented runtime directory.",
            "Keep prose paragraphs and list items on one physical line; rely on editor soft wrapping.",
            prompt,
        ]
        for path in context or []:
            if path.exists():
                label = f"{path.parent.name}/{path.name}"
                sections.append(f"--- BEGIN CONTEXT: {label} ---\n{path.read_text()}\n--- END CONTEXT: {label} ---")
        effective_prompt = "\n\n".join(sections)
        started = time.perf_counter()
        usage: dict[str, int] = {}
        error = None
        warnings = []
        terminal_state = None
        returncode = None
        # A directory outside the repository avoids incidental project/source/gold context.
        with tempfile.TemporaryDirectory(prefix="decompbench-codex-") as temporary:
            workspace = Path(temporary).resolve()
            staged = workspace / target.name
            staged.write_text(target.read_text() if target.exists() else "")
            effective_prompt = record_workspace_prompt(workspace, target.name, effective_prompt, log_dir, step)
            command = [
                executable, "--no-daemon", "exec", "--ephemeral", "--ignore-user-config",
                "--ignore-rules", "--skip-git-repo-check", "--sandbox", "workspace-write", "--json",
                "--color", "never", "--cd", str(workspace),
                "--disable", "plugins", "--disable", "remote_plugin", "--disable", "apps",
                "--disable", "hooks", "--disable", "memories", "--disable", "multi_agent",
                "--disable", "skill_search", "--enable", "skip_host_skill_discovery",
                "--disable", "shell_snapshot",
                "-c", "project_doc_max_bytes=0", "-c", 'developer_instructions=""',
                "-c", 'web_search="disabled"', "-c", 'approval_policy="never"',
                "-c", 'shell_environment_policy.inherit="none"',
                "-c", "shell_environment_policy.set={PATH=" + json.dumps(os.environ.get("PATH", "")) + "}",
                "--output-last-message", str(workspace / "response.md"),
            ]
            if self.model:
                command.extend(["--model", self.model])
            if self.effort:
                command.extend(["-c", "model_reasoning_effort=" + json.dumps(self.effort)])
            command.append("-")
            # Record configuration without host paths or environment values.
            (log_dir / f"{step}_invocation.json").write_text(json.dumps({
                "backend": "codex-cli", "requested_model": self.model,
                "requested_effort": self.effort, "role": self.role,
                "ephemeral": True, "ignore_user_config": True, "project_doc_max_bytes": 0,
                "sandbox": "workspace-write", "external_context": False,
                "target": target.name, "timeout_seconds": self.timeout,
            }, indent=2))
            with (log_dir / f"{step}_events.jsonl").open("w") as events, (log_dir / f"{step}_stderr.log").open("w") as stderr:
                process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=events, stderr=stderr, text=True, start_new_session=True)
                try:
                    process.communicate(input=effective_prompt, timeout=self.timeout)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    error = f"Codex edit timed out after {self.timeout} seconds"
                returncode = process.returncode
            for line in (log_dir / f"{step}_events.jsonl").read_text().splitlines():
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event.get("type") == "turn.completed":
                    terminal_state = "completed"
                    if returncode == 0:
                        error = None
                    for key, value in event.get("usage", {}).items():
                        if isinstance(value, int):
                            usage[key] = usage.get(key, 0) + value
                if event.get("type") in {"error", "turn.failed"}:
                    payload = event.get("error") or event
                    message = str(payload.get("message", payload)) if isinstance(payload, dict) else str(payload)
                    if event["type"] == "turn.failed":
                        terminal_state = "failed"
                        error = message
                    else:
                        warnings.append(message)
            infrastructure_markers = ("code-mode host exited during handshake", "timed out negotiating with the code-mode host", "failed to initialize code-mode", "failed to spawn code-mode")
            diagnostics = (log_dir / f"{step}_stderr.log").read_text()
            host_failure = next((marker for marker in infrastructure_markers if marker in diagnostics.lower()), None)
            if host_failure:
                # A terminal model response can still follow a failed local tool host.
                # Such a response must not turn an OS launch gate into a model failure.
                error = "Local Codex infrastructure failure: " + host_failure
                terminal_state = "infrastructure_failed"
            if (workspace / "response.md").exists():
                shutil.copyfile(workspace / "response.md", log_dir / f"{step}_response.md")
            ok = returncode == 0 and error is None and terminal_state == "completed" and staged.exists() and bool(staged.read_text().strip())
            if returncode == 0 and error is None and terminal_state == "completed" and not ok:
                record_wrong_target(workspace, target.name, log_dir, step)
            if ok:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(staged, target)
            elif error is None:
                error = warnings[-1] if warnings else f"Codex did not complete a successful edit of {target.name} (status {returncode}); see {step}_stderr.log"
        result = CodexResult(ok, returncode, time.perf_counter() - started, usage, error, warnings, terminal_state, hashlib.sha256(target.read_bytes()).hexdigest() if ok else None)
        self._save_result(log_dir, step, result)
        return result

    @staticmethod
    def _save_result(log_dir: Path, step: str, result: CodexResult) -> None:
        path = log_dir / f"{step}_metrics.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(asdict(result), indent=2))
        temporary.replace(path)
