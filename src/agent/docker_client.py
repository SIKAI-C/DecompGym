"""Fresh Docker CLI edits with private credentials and recoverable call records."""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
import threading

from src.agent.cli_events import parse_events
from src.agent.result import AgentResult, save_result
from src.agent.container import DockerRuntime
from src.agent.prompts.policy import WORKSPACE_POLICY, record_workspace_prompt
from src.agent.safe_artifacts import read_regular_text
from src.experiments.environment import RuntimeEnvironmentError


DOCKER_EDIT_POLICY = "fresh-docker-cli-edit-v1"
IMAGE_TAG = "decompbench-agents-v2:20261013"
CLI_VERSIONS = {"codex-cli": "codex-cli 0.162.1", "claude-code-cli": "2.1.296 (Claude Code)", "agy-cli": "1.3.1"}
AUTH_VOLUMES = {"codex-cli": "decompbench-v2-codex-auth", "claude-code-cli": "decompbench-v2-claude-auth", "agy-cli": "decompbench-agy-native-keyring-v1"}
_PREFLIGHT_LOCK = threading.Lock()
_PREFLIGHTS = set()


def worker_file_hashes():
    root = Path(__file__).resolve().parents[2]
    paths = [root / "docker/agents/worker.py", root / "docker/agents/agy_credentials.py", root / "src/agent/safe_artifacts.py", root / "src/agent/cli_events.py"]
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


class AuthenticationRequired(RuntimeEnvironmentError):
    """The owner must supply subscription authentication through a private channel."""


def credential_from_environment(backend):
    if backend == "codex-cli":
        directory = Path.home() / ".codex"
        value = read_regular_text(directory, "auth.json", missing_ok=True)
        return json.loads(value) if value else None
    if backend == "claude-code-cli":
        token = os.environ.get("CLAUDE_CODE_OAUTH_TOKEN")
        if token:
            return {"oauth_token": token}
        directory = Path.home() / ".claude"
        value = read_regular_text(directory, ".credentials.json", missing_ok=True) if directory.exists() else None
        return {"credentials": json.loads(value)} if value else None
    password = os.environ.get("DECOMPBENCH_AGY_KEYRING_PASSPHRASE")
    if not password:
        raise AuthenticationRequired("AGY requires its benchmark keyring passphrase in the local launch environment; do not send it in chat")
    return {"keyring_passphrase": password}


class DockerCliClient:
    def __init__(self, model, *, backend, image_id, runtime: DockerRuntime, evidence_root: Path, effort="medium", timeout=900, role="generation", credential_provider=credential_from_environment):
        if backend not in CLI_VERSIONS or not image_id.startswith("sha256:"):
            raise ValueError("An approved backend and immutable image ID are required")
        self.model, self.backend, self.image_id = model, backend, image_id
        self.runtime, self.evidence_root = runtime, evidence_root
        self.effort, self.timeout, self.role = effort, timeout, role
        self.credential_provider = credential_provider

    def protocol(self):
        return {"backend": self.backend, "transport": "docker", "transport_policy": DOCKER_EDIT_POLICY, "model": self.model, "effort": self.effort, "role": self.role,
                "timeout": self.timeout, "workspace_policy": WORKSPACE_POLICY, "image_id": self.image_id, "cli_version": CLI_VERSIONS[self.backend]}

    def prepare(self):
        key = (self.image_id, hashlib.sha256(self.runtime.seccomp_profile.read_bytes()).hexdigest())
        with _PREFLIGHT_LOCK:
            if key not in _PREFLIGHTS:
                directory = self.evidence_root / "image-preflight" / self.image_id.split(":", 1)[1]
                result = self.runtime.run(self.image_id, ["preflight"], state_path=directory / "container.json", timeout=45, artifacts=["worker.json"])
                observed = json.loads(result.artifacts.get("worker.json", "{}"))
                expected = {"codex": CLI_VERSIONS["codex-cli"], "claude": CLI_VERSIONS["claude-code-cli"], "agy": CLI_VERSIONS["agy-cli"]}
                if result.returncode or not observed.get("ok") or observed.get("versions") != expected or observed.get("worker_files_sha256") != worker_file_hashes():
                    raise RuntimeEnvironmentError("Docker isolation or CLI version preflight failed before authentication")
                _PREFLIGHTS.add(key)
        volume = AUTH_VOLUMES[self.backend]
        if self.backend == "agy-cli":
            self.runtime.ensure_volume(volume, "agy-native-auth", label="io.decompbench.purpose", create=False)
            return {volume: "/auth/keyrings"}
        self.runtime.ensure_volume(volume, self.backend + "-v2-auth")
        return {volume: "/auth"}

    def compose_prompt(self, target_name, prompt, context=None):
        """Workspace rules, the task prompt and each context file; record_workspace_prompt then rewrites the first line."""
        sections = [f"Edit only {target_name}. All task context is supplied below.", "Do not inspect other directories, use external tools or services, or search for reference implementations.",
                    "Do not execute the generated solver. The harness executes it later in its documented runtime directory.", "Keep prose paragraphs and list items on one physical line; rely on editor soft wrapping.", prompt]
        if self.backend == "agy-cli":
            sections[1:4] = ["Do not inspect other directories, use external tools or services, or search for reference implementations. Do not execute the generated solver. Do not use memory, skills, subagents or prior conversations. Keep prose paragraphs on one physical line."]
        for path in context or []:
            if path.exists():
                contents = read_regular_text(path.parent, path.name)
                label = f"{path.parent.name}/{path.name}"
                ending = "--- END CONTEXT ---" if self.backend == "agy-cli" else f"--- END CONTEXT: {label} ---"
                sections.append(f"--- BEGIN CONTEXT: {label} ---\n{contents}\n{ending}")
        return "\n\n".join(sections)

    def edit_file(self, target, prompt, *, context=None, log_dir=None, step="agent"):
        target = Path(target).absolute()
        log_dir = Path(log_dir or target.parent).resolve()
        log_dir.mkdir(parents=True, exist_ok=True)
        if target.is_symlink() or target.name not in {"solver.py", "debug.md", "alg_verifier.md"}:
            raise ValueError("Docker edits require an ordinary declared target")
        effective = record_workspace_prompt(Path("/work"), target.name, self.compose_prompt(target.name, prompt, context), log_dir, step, old_absolute_note=self.backend == "agy-cli")
        mounts = self.prepare()
        credential = self.credential_provider(self.backend)
        job = {"target": target.name, "initial_content": read_regular_text(target.parent, target.name, missing_ok=True) or "", "prompt": effective,
               "model": self.model, "effort": self.effort, "role": self.role, "timeout": self.timeout, "credential": credential}
        result = self.runtime.run(self.image_id, ["agent", self.backend], payload=job, state_path=log_dir / f"{step}_container.json", timeout=self.timeout, network="bridge", volumes=mounts,
                                  artifacts=[target.name, "events.jsonl", "stderr.log", "response.md", "worker.json", "delivery.json"], public_metadata={"backend": self.backend, "target": target.name, "model": self.model, "role": self.role})
        return self._record(target, log_dir, step, result)

    def recover_file(self, target, *, log_dir, step):
        state = log_dir / f"{step}_container.json"
        if not state.exists():
            return None
        return self._record(target, log_dir, step, self.runtime.recover(state))

    def _record(self, target, log_dir, step, result):
        for name in ("events.jsonl", "stderr.log", "response.md", "worker.json", "delivery.json"):
            if name in result.artifacts:
                (log_dir / f"{step}_{name}").write_text(result.artifacts[name])
        worker = json.loads(result.artifacts.get("worker.json", "{}"))
        returncode = worker.get("returncode", result.returncode)
        parsed = parse_events(self.backend, result.artifacts.get("events.jsonl", ""), returncode=returncode, requested_model=self.model)
        if worker:
            if worker.get("cli_version") != CLI_VERSIONS[self.backend] or worker.get("requested_model") != self.model or worker.get("target") != target.name:
                raise RuntimeEnvironmentError("Docker worker identity differs from the frozen invocation")
            parsed.update({key: worker[key] for key in ("terminal_state", "error", "usage", "usage_available", "warnings", "observed_models") if key in worker})
        error, terminal = parsed["error"], parsed["terminal_state"]
        if self.backend == "claude-code-cli" and terminal == "completed":
            observed = parsed["observed_models"]
            if not observed or any(model != self.model and re.fullmatch(re.escape(self.model) + r"-\d{8}", model) is None for model in observed):
                error, terminal = f"Claude Code model identity unavailable or mismatched: requested {self.model}, observed {observed}", "infrastructure_failed"
        if result.timed_out or result.output_limited or result.oom_killed:
            error, terminal = "Docker invocation interrupted by its resource or time limit", "infrastructure_failed"
        diagnostics = result.artifacts.get("stderr.log", "").lower()
        host_markers = ("code-mode host exited during handshake", "timed out negotiating with the code-mode host", "failed to initialize code-mode", "failed to spawn code-mode")
        if any(marker in diagnostics for marker in host_markers):
            error, terminal = "CLI tool host failed to initialize", "infrastructure_failed"
        contents = result.artifacts.get(target.name, "")
        ok = returncode == 0 and terminal == "completed" and error is None and bool(contents.strip())
        if ok:
            target.write_text(contents)
        elif not target.exists():
            target.write_text("")
        delivery = json.loads(result.artifacts.get("delivery.json", "{}"))
        misplaced = []
        for relative, text in delivery.get("misplaced", {}).items():
            path = Path(relative)
            if path.is_absolute() or ".." in path.parts or path.name != target.name:
                raise RuntimeEnvironmentError("Unexpected Docker artifact path")
            archive = log_dir / "unexpected_outputs" / path
            archive.parent.mkdir(parents=True, exist_ok=True)
            archive.write_text(text)
            misplaced.append({"relative_path": relative, "archive": str(archive.relative_to(log_dir)), "sha256": hashlib.sha256(text.encode()).hexdigest()})
        if misplaced:
            workspace_record = log_dir / f"{step}_workspace.json"
            record = {"kind": "wrong_target_file", "workspace_note_version": WORKSPACE_POLICY, "target_name": target.name, "required_target_empty": True,
                      "workspace_record_sha256": hashlib.sha256(workspace_record.read_bytes()).hexdigest(), "misplaced_files": misplaced}
            (log_dir / f"{step}_delivery_failure.json").write_text(json.dumps(record, indent=2) + "\n")
        invocation = self.protocol() | result.metadata | {"observed_models": parsed["observed_models"], "usage_available": parsed["usage_available"], "requested_model": self.model,
                                                          "target": target.name, "terminal_state": terminal, "ephemeral": True, "external_context": False, "candidate_rejection": delivery.get("rejected_target")}
        (log_dir / f"{step}_invocation.json").write_text(json.dumps(invocation, indent=2) + "\n")
        answer = AgentResult(ok, returncode, worker.get("agent_seconds", result.elapsed_seconds), parsed["usage"], error if error else (None if ok else "Completed CLI did not deliver the requested file"), parsed["warnings"], terminal,
                             hashlib.sha256(contents.encode()).hexdigest() if ok else None)
        save_result(log_dir, step, answer)
        if worker.get("credential_checkpoint_error"):
            raise RuntimeEnvironmentError("The completed CLI artifact was saved, but credential persistence needs repair")
        if not worker and "worker_error" in result.stdout:
            raise AuthenticationRequired(f"{self.backend} could not initialize its private authentication channel; inspect the saved container diagnostics")
        return answer
