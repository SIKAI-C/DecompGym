"""Fresh, prompt-scoped file editing through the locally installed Codex CLI."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import tempfile
import time
from typing import Any


@dataclass
class CodexResult:
    ok: bool
    returncode: int | None
    elapsed_seconds: float
    usage: dict[str, int]
    error: str | None = None
    warnings: list[str] = field(default_factory=list)


class CodexClient:
    """Start one fresh CLI session per edit and copy back only the requested file."""

    def __init__(self, model: str | None = None, *, executable: str = "codex", timeout: int = 900) -> None:
        self.model = model
        self.executable = executable
        self.timeout = timeout

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
        (log_dir / f"{step}_prompt.md").write_text(effective_prompt)
        started = time.perf_counter()
        usage: dict[str, int] = {}
        error = None
        warnings = []
        terminal_state = None
        returncode = None
        # A directory outside the repository avoids incidental project/source/gold context.
        with tempfile.TemporaryDirectory(prefix="decompbench-codex-") as temporary:
            workspace = Path(temporary)
            staged = workspace / target.name
            staged.write_text(target.read_text() if target.exists() else "")
            command = [
                executable, "--no-daemon", "exec", "--ephemeral", "--ignore-user-config",
                "--skip-git-repo-check", "--sandbox", "workspace-write", "--json",
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
            command.append("-")
            # Record configuration without host paths or environment values.
            (log_dir / f"{step}_invocation.json").write_text(json.dumps({
                "backend": "codex-cli", "requested_model": self.model,
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
            if (workspace / "response.md").exists():
                shutil.copyfile(workspace / "response.md", log_dir / f"{step}_response.md")
            ok = returncode == 0 and error is None and terminal_state == "completed" and staged.exists() and bool(staged.read_text().strip())
            if ok:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(staged, target)
            elif error is None:
                error = warnings[-1] if warnings else f"Codex did not complete a successful edit of {target.name} (status {returncode}); see {step}_stderr.log"
        result = CodexResult(ok, returncode, time.perf_counter() - started, usage, error, warnings)
        self._save_result(log_dir, step, result)
        return result

    @staticmethod
    def _save_result(log_dir: Path, step: str, result: CodexResult) -> None:
        (log_dir / f"{step}_metrics.json").write_text(json.dumps(asdict(result), indent=2))
