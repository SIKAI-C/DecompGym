"""Fresh Docker CLI edits with private credentials and recoverable call records."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import os
import re
from pathlib import Path
import threading

from src.agent.auth import AUTH_VOLUMES, auth_volume, forbid_job_credential, mounted_auth_volume, repair_command, require_exclusive_volume, terminal_auth_marker, volume_lock
from src.agent.cli_events import parse_events
from src.agent.result import AgentResult, save_result
from src.agent.container import DockerRuntime
from src.agent import mounted_context
from src.agent.prompts.mounted import (CONTEXT_DELIVERIES, MOUNTED_CPP_WORKSPACE_POLICY, MOUNTED_REVIEW_WORKSPACE_POLICY, MOUNTED_WORKSPACE_POLICY, REVIEW_HEADING, REVIEW_TARGET, WORKSPACE,
                                      review_files, review_workspace_part, workspace_part)
from src.agent.prompts.policy import WORKSPACE_POLICY, record_workspace_prompt
from src.agent.safe_artifacts import read_regular_text
from src.agent.specs import DEFAULT_RUNTIME, MOUNT_LAYOUT, REVIEW_LAYOUT, SINGLE_FILE_DELIVERY, MountLayout, RuntimeSpec
from src.experiments.environment import RuntimeEnvironmentError


DOCKER_EDIT_POLICY = "fresh-docker-cli-edit-v1"
IMAGE_TAG = "decompbench-agents-v2:20261013"
CLI_VERSIONS = {"codex-cli": "codex-cli 0.162.1", "claude-code-cli": "2.1.296 (Claude Code)", "agy-cli": "1.3.1"}
_PREFLIGHT_LOCK = threading.Lock()
_PREFLIGHTS = set()


def worker_file_hashes():
    root = Path(__file__).resolve().parents[2]
    paths = [root / "docker/agents/worker.py", root / "docker/agents/agy_credentials.py", root / "src/agent/safe_artifacts.py", root / "src/agent/cli_events.py"]
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


class AuthenticationRequired(RuntimeEnvironmentError):
    """The owner must repair or supply subscription authentication through a private channel (exit 64)."""

    def __init__(self, message, evidence=None):
        super().__init__(message)
        self.evidence = evidence or {}
        self.raised_at = datetime.now(timezone.utc).isoformat()  # when the failure was observed (sentinel lifecycle)


def credential_from_environment(backend):
    """The job credential. None for Codex and Claude Code: their auth volume is the only source (src/agent/auth.py),
    so neither the host's ~/.codex/auth.json nor a launch-environment CLAUDE_CODE_OAUTH_TOKEN is used unless
    `docker auth --login` imported it explicitly. AGY receives only its keyring passphrase."""
    if backend in ("codex-cli", "claude-code-cli"):
        return None
    password = os.environ.get("DECOMPBENCH_AGY_KEYRING_PASSPHRASE")
    if not password:
        raise AuthenticationRequired("AGY requires its benchmark keyring passphrase in the local launch environment; do not send it in chat")
    return {"keyring_passphrase": password}


class DockerCliClient:
    # The writable target of every call is the one requested file (src.agent.specs); multi-file delivery is MultiFileDockerCliClient.
    delivery = SINGLE_FILE_DELIVERY

    def __init__(self, model, *, backend, image_id, runtime: DockerRuntime, evidence_root: Path, effort="medium", timeout=900, role="generation", credential_provider=credential_from_environment,
                 context_delivery=None, context_layout: MountLayout | None = None, runtime_spec: RuntimeSpec | None = None):
        if backend not in CLI_VERSIONS or not image_id.startswith("sha256:"):
            raise ValueError("An approved backend and immutable image ID are required")
        if context_delivery not in CONTEXT_DELIVERIES:
            raise ValueError(f"Unknown context delivery: {context_delivery!r}")
        if context_delivery is None and (context_layout is not None or runtime_spec is not None):
            raise ValueError("A context layout and a runtime contract are part of the mounted-files prompt only")
        self.model, self.backend, self.image_id = model, backend, image_id
        self.runtime, self.evidence_root = runtime, evidence_root
        self.effort, self.timeout, self.role = effort, timeout, role
        self.credential_provider = credential_provider
        # The call-gate slot whose Codex or Claude Code auth volume this call uses (None: slot 0); see bind_auth_slot.
        self.auth_slot = None
        # None: the paper's pasted context (compose_prompt). CONTEXT_DELIVERY_MOUNTED: read-only files in the workspace,
        # placed by `context_layout` and described with the run-time contract of `runtime_spec` (src.agent.specs).
        self.context_delivery = context_delivery
        self.context_layout = context_layout or MOUNT_LAYOUT
        self.runtime_spec = runtime_spec or DEFAULT_RUNTIME

    def protocol(self):
        value = {"backend": self.backend, "transport": "docker", "transport_policy": DOCKER_EDIT_POLICY, "model": self.model, "effort": self.effort, "role": self.role,
                 "timeout": self.timeout, "workspace_policy": WORKSPACE_POLICY, "image_id": self.image_id, "cli_version": CLI_VERSIONS[self.backend]}
        if self.context_delivery is not None:
            value["context_delivery"] = self.context_delivery
        return value

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
        return self.auth_mounts()

    def auth_volume(self):
        return auth_volume(self.backend, self.auth_slot)

    @contextmanager
    def bind_auth_slot(self, slot):
        """Use the auth volume of call-gate slot `slot` (Codex; Claude Code in the lanes variant) while the gate holds that slot."""
        previous, self.auth_slot = self.auth_slot, slot
        try:
            yield
        finally:
            self.auth_slot = previous

    def auth_mounts(self):
        volume = self.auth_volume()
        if self.backend == "agy-cli":
            self.runtime.ensure_volume(volume, "agy-native-auth", label="io.decompbench.purpose", create=False)
            return {volume: "/auth/keyrings"}
        self.runtime.ensure_volume(volume, self.backend + "-v2-auth")
        return {volume: "/auth"}

    @contextmanager
    def auth_lock(self, mounts, state_path=None):
        """Hold the host lock of the auth volume a container of this call may use, until the call returns.

        An existing container state (a recovery, also inside DockerRuntime.run) is locked on the volume its container
        mounted, which may be another slot's than the one bound now. With the lock held, no other running container
        may still use the volume (require_exclusive_volume)."""
        volumes = list(mounts or {})
        if len(volumes) > 1:
            raise RuntimeEnvironmentError("A Docker agent call mounts exactly one auth volume")
        recorded, allowed = None, ()
        if state_path is not None and Path(state_path).exists():
            try:
                state = json.loads(Path(state_path).read_text())
            except (OSError, ValueError):
                state = {}
            recorded = mounted_auth_volume(state)
            allowed = (state["name"],) if isinstance(state.get("name"), str) else ()
        volume = recorded or (volumes[0] if volumes else None)
        if volume is None:
            yield None
            return
        with volume_lock(volume):
            require_exclusive_volume(self.runtime, volume, allowed=allowed)
            yield volume

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

    def compose_mounted_prompt(self, target_name, prompt, files):
        """Part 1 (workspace and files) before the runner's structured task prompt; returns (prompt, workspace note). A
        method-judge call (alg_verifier.md, REVIEW_LAYOUT only) gets the review Part 1 and the review text as Part 2."""
        if (target_name == REVIEW_TARGET) != (self.context_layout == REVIEW_LAYOUT):
            raise ValueError("Only a method-judge call edits alg_verifier.md, and only with the review layout")
        if target_name == REVIEW_TARGET:
            workspace, note = review_workspace_part(target_name, files, self.backend, delivery=self.delivery, layout=self.context_layout)
            return workspace + "\n\n" + REVIEW_HEADING + "\n\n" + prompt, note
        workspace, note = workspace_part(target_name, files, self.backend, delivery=self.delivery, runtime=self.runtime_spec, layout=self.context_layout)
        return workspace + "\n\n" + prompt, note

    def mounted_prompt(self, target, prompt, context, log_dir, step):
        """The mounted-files part of a call (single- or multi-file): (effective prompt, DockerRuntime.run context
        arguments). Records the workspace prompt and the context evidence ({step}_context.json)."""
        files = review_files(context, self.context_layout) if target.name == REVIEW_TARGET and self.context_layout == REVIEW_LAYOUT else mounted_context.mounted_files(context, self.context_layout)
        composed, note = self.compose_mounted_prompt(target.name, prompt, files)
        version = MOUNTED_REVIEW_WORKSPACE_POLICY if target.name == REVIEW_TARGET else MOUNTED_CPP_WORKSPACE_POLICY if self.delivery.sources(target.name) else MOUNTED_WORKSPACE_POLICY
        effective = record_workspace_prompt(Path(WORKSPACE), target.name, composed, log_dir, step, old_absolute_note=self.backend == "agy-cli", note=note, version=version)
        (log_dir / f"{step}_context.json").write_text(json.dumps(mounted_context.evidence(files, target.name, self.delivery, self.context_layout), indent=2) + "\n")
        run_context = {"context": mounted_context.volume_contents(files)}
        if self.context_layout != MOUNT_LAYOUT:
            run_context["context_directories"] = self.context_layout.directories
        return effective, run_context

    def edit_file(self, target, prompt, *, context=None, log_dir=None, step="agent"):
        target = Path(target).absolute()
        log_dir = Path(log_dir or target.parent).resolve()
        log_dir.mkdir(parents=True, exist_ok=True)
        if target.is_symlink() or target.name not in {"solver.py", "debug.md", "alg_verifier.md"}:
            raise ValueError("Docker edits require an ordinary declared target")
        if self.context_delivery is not None:
            return self._edit_mounted(target, prompt, context, log_dir, step)
        effective = record_workspace_prompt(Path("/work"), target.name, self.compose_prompt(target.name, prompt, context), log_dir, step, old_absolute_note=self.backend == "agy-cli")
        mounts = self.prepare()
        credential = self.credential_provider(self.backend)
        forbid_job_credential(self.backend, credential)
        job = {"target": target.name, "initial_content": read_regular_text(target.parent, target.name, missing_ok=True) or "", "prompt": effective,
               "model": self.model, "effort": self.effort, "role": self.role, "timeout": self.timeout, "credential": credential}
        with self.auth_lock(mounts, log_dir / f"{step}_container.json"):
            result = self.runtime.run(self.image_id, ["agent", self.backend], payload=job, state_path=log_dir / f"{step}_container.json", timeout=self.timeout, network="bridge", volumes=mounts,
                                      artifacts=[target.name, "events.jsonl", "stderr.log", "response.md", "worker.json", "delivery.json"], public_metadata={"backend": self.backend, "target": target.name, "model": self.model, "role": self.role})
        return self._record(target, log_dir, step, result)

    def _edit_mounted(self, target, prompt, context, log_dir, step):
        """The mounted-files call: the same job and collection as the paper path, with the context files mounted read-only."""
        effective, run_context = self.mounted_prompt(target, prompt, context, log_dir, step)
        mounts = self.prepare()
        credential = self.credential_provider(self.backend)
        forbid_job_credential(self.backend, credential)
        job = {"target": target.name, "initial_content": read_regular_text(target.parent, target.name, missing_ok=True) or "", "prompt": effective,
               "model": self.model, "effort": self.effort, "role": self.role, "timeout": self.timeout, "credential": credential}
        with self.auth_lock(mounts, log_dir / f"{step}_container.json"):
            result = self.runtime.run(self.image_id, ["agent", self.backend], payload=job, state_path=log_dir / f"{step}_container.json", timeout=self.timeout, network="bridge", volumes=mounts,
                                      artifacts=[target.name, "events.jsonl", "stderr.log", "response.md", "worker.json", "delivery.json"], public_metadata={"backend": self.backend, "target": target.name, "model": self.model, "role": self.role,
                                                                                                                                             "context_delivery": self.context_delivery},
                                      **run_context)
        return self._record(target, log_dir, step, result)

    def _ignore_context_copies(self, result, log_dir, step):
        """The worker's search for a misplaced target walks /work, so it also meets the read-only mounted copies (for
        example /work/history/attempt_01/solver.py). They are context, never a delivery: drop them from the delivery
        report and record what was dropped. Only the mounted-files protocol has such copies."""
        if self.context_delivery is None or "delivery.json" not in result.artifacts:
            return result
        delivery = json.loads(result.artifacts["delivery.json"])
        prefixes = tuple(directory[len(WORKSPACE) + 1:] + "/" for directory in self.context_layout.directories)
        misplaced = delivery.get("misplaced", {})
        ignored = sorted(path for path in misplaced if path.startswith(prefixes))
        if not ignored:
            return result
        delivery["misplaced"] = {path: text for path, text in misplaced.items() if path not in ignored}
        (log_dir / f"{step}_context_copies_ignored.json").write_text(json.dumps({"ignored_mounted_copies": ignored}, indent=2) + "\n")
        return replace(result, artifacts=result.artifacts | {"delivery.json": json.dumps(delivery)})

    def recover_file(self, target, *, log_dir, step):
        state = log_dir / f"{step}_container.json"
        if not state.exists():
            return None
        # A recovered container may still run and write its auth volume: hold the lock of the volume it mounted.
        with self.auth_lock(None, state):
            result = self.runtime.recover(state)
        return self._record(target, log_dir, step, result)

    def _record(self, target, log_dir, step, result):
        result = self._ignore_context_copies(result, log_dir, step)
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
        # A terminal credential failure is infrastructure: the call is recorded but never committed, so it is retried
        # after the owner repairs the credential and is never charged to the agent.
        auth_marker = terminal_auth_marker(self.backend, result.artifacts.get("events.jsonl", ""), result.artifacts.get("stderr.log", ""), succeeded=returncode == 0 and terminal == "completed" and error is None)
        if auth_marker:
            error, terminal = f"credential_invalid: {auth_marker}", "infrastructure_failed"
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
        volume = mounted_auth_volume(result.metadata) or self.auth_volume()
        if auth_marker:
            evidence = {"kind": "credential_invalid", "backend": self.backend, "auth_volume": volume, "marker": auth_marker, "role": self.role, "model": self.model,
                        "call": str(log_dir), "step": step, "recorded_at": datetime.now(timezone.utc).isoformat(), "committed": False, "charged_to_agent": False,
                        "repair": repair_command(self.backend, volume)}
            (log_dir / f"{step}_auth_invalid.json").write_text(json.dumps(evidence, indent=2) + "\n")
            raise AuthenticationRequired(f"{self.backend} credential in {volume} is invalid ({auth_marker}); repair with `{evidence['repair']}`", evidence)
        if worker.get("credential_checkpoint_error"):
            raise RuntimeEnvironmentError("The completed CLI artifact was saved, but credential persistence needs repair")
        if not worker and "worker_error" in result.stdout:
            raise AuthenticationRequired(f"{self.backend} could not initialize its private authentication channel; inspect the saved container diagnostics (repair: `{repair_command(self.backend, volume)}`)",
                                         {"kind": "credential_unavailable", "backend": self.backend, "auth_volume": volume, "call": str(log_dir), "step": step, "repair": repair_command(self.backend, volume)})
        return answer
