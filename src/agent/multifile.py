"""Optional multi-file delivery for the large-scale PDPTW C++ track (protocol v4 §5 "Allowed sources" / "Delivery").

Single-file delivery (``solver.py`` only, `DockerCliClient`, the paper image) is the default for every task and is not changed by
this module. Multi-file delivery is a separate, explicit client (`MultiFileDockerCliClient`) that runs only when the task asks for
it through `delivery_spec`, whose registry `MULTI_FILE_TASKS` admits only the large-scale PDPTW tracks.

Contract. The agent edits ``solver.py`` and, in the ``cpp_allowed`` track, may create flat ``cpp/*.cpp`` / ``cpp/*.hpp`` (C++17; at
most 64 files, 2 MiB per file, 8 MiB in total); the ``python_only`` track keeps the paper's instruction and rejects any C++ file.
Inside the container, ``docker/agents/multifile_worker.py`` validates the workspace (no absolute paths, ``..``, links, special
files, nested directories or other extensions) and returns a hashed manifest (path, size, sha256, sorted by path) with the files.
Here the delivery is re-validated with the Phoenix-side canonical validator (`xl_build.manifest_for` / `verify_manifest`), so the
manifest is exactly what `xl_executor.XLExecutor.execute((manifest, files), ...)` consumes. The committed delivery is
``solver.py`` plus ``cpp/`` next to the target and ``{step}_delivery_manifest.json`` in the call directory; `load_delivery` reads
it back verified. A delivery that violates the contract is a completed generation without a usable file, like an empty
``solver.py`` in the paper path.

Image. The paper image is unchanged and has no multi-file entry. The multi-file image (`MULTIFILE_IMAGE_TAG`, built from
``docker/agents/Dockerfile.multifile``) is the paper image plus ``/opt/decompbench/multifile_worker.py``; its preflight reports
the paper worker hashes and the extension hash, and both must match this checkout.
"""
from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import threading

from src.agent.container import DockerRuntime
from src.agent.docker_client import AUTH_VOLUMES, CLI_VERSIONS, DockerCliClient, worker_file_hashes
from src.agent.prompts.policy import WORKSPACE_POLICY, edit_instruction, record_workspace_prompt
from src.agent.result import save_result
from src.agent.safe_artifacts import UnsafeArtifact, read_regular, read_regular_text
from src.experiments.environment import RuntimeEnvironmentError
from src.experiments.xl_build import MAX_FILE_BYTES, TRACKS, DeliveryError, manifest_for, verify_manifest

SINGLE_FILE, MULTI_FILE = "single_file", "multi_file"
MULTIFILE_POLICY = "fresh-docker-cli-multifile-v1"
MULTIFILE_IMAGE_TAG = "decompgym-agents-multifile-v1:20261105"
EXTENSION_ENTRY = "/opt/decompbench/multifile_worker.py"
BUNDLE_NAME = "delivery_bundle.json"
RECORD_VERSION = "decompgym-multifile-worker-v1"
CPP_WORKSPACE_POLICY = "existing-target-cwd-cpp-v1"
CPP_NOTE = ("You may also create C++17 source files as flat cpp/*.cpp and cpp/*.hpp files under the current working directory "
            "(at most 64 C++ files, 2 MiB per file, 8 MiB in total including solver.py); create no other files or directories.")
# (problem, scale) -> tracks that use multi-file delivery. Every other task is single-file only.
MULTI_FILE_TASKS = {("pdptw", "xl"): ("python_only", "cpp_allowed")}
_PREFLIGHT_LOCK = threading.Lock()
_PREFLIGHTS = set()


@dataclass(frozen=True)
class DeliverySpec:
    mode: str = SINGLE_FILE
    track: str | None = None

    def __post_init__(self):
        if self.mode == SINGLE_FILE and self.track is not None:
            raise ValueError("single-file delivery has no track")
        if self.mode == MULTI_FILE and self.track not in TRACKS:
            raise ValueError(f"multi-file delivery requires a track in {TRACKS}")
        if self.mode not in (SINGLE_FILE, MULTI_FILE):
            raise ValueError(f"unknown delivery mode {self.mode!r}")


SINGLE_FILE_DELIVERY = DeliverySpec()


def delivery_spec(problem: str, scale: str, track: str | None = None) -> DeliverySpec:
    """The task's delivery: single-file unless the task explicitly requests a track that `MULTI_FILE_TASKS` admits."""
    if track is None:
        return SINGLE_FILE_DELIVERY
    allowed = MULTI_FILE_TASKS.get((problem, scale), ())
    if track not in allowed:
        raise ValueError(f"multi-file delivery is not enabled for {problem}/{scale} track {track!r}")
    return DeliverySpec(MULTI_FILE, track)


def extension_file_hashes() -> dict[str, str]:
    path = Path(__file__).resolve().parents[2] / "docker/agents/multifile_worker.py"
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest()}


class MultiFileDockerRuntime(DockerRuntime):
    """The paper lifecycle, entering the multi-file worker. The collector ceiling covers the bundle (8 MiB of sources, about
    10.7 MiB as base64) next to the paper worker's logs (4 MiB each, up to six times larger when JSON-escaped)."""

    WORKER = EXTENSION_ENTRY
    COLLECT_OUTPUT_BYTES = 128 * 1024 * 1024


def verify_bundle(text: str, solver: str, track: str) -> tuple[dict, dict[str, bytes]]:
    """Re-validate the worker's delivery with the canonical validator; return (manifest, files) or raise DeliveryError."""
    try:
        return _verify_bundle(text, solver, track)
    except DeliveryError:
        raise
    except (TypeError, KeyError, AttributeError, ValueError) as error:
        raise DeliveryError("malformed delivery record") from error


def _verify_bundle(text, solver, track):
    try:
        value = json.loads(text)
    except ValueError as error:
        raise DeliveryError("unreadable delivery record") from error
    if not isinstance(value, dict) or value.get("version") != RECORD_VERSION or value.get("mode") != MULTI_FILE or value.get("track") != track:
        raise DeliveryError("delivery record has another version, mode or track")
    if value.get("error") is not None or value.get("manifest") is None:
        raise DeliveryError(f"delivery rejected: {value.get('error') or 'no manifest'}", value.get("error_rejected") or [])
    manifest, encoded = value["manifest"], value.get("files")
    if not isinstance(manifest, dict) or not isinstance(encoded, dict) or manifest.get("track") != track:
        raise DeliveryError("malformed delivery manifest")
    entries = manifest.get("files")
    listed = [entry.get("path") for entry in entries if isinstance(entry, dict)] if isinstance(entries, list) else []
    if not isinstance(entries, list) or len(listed) != len(entries) or not all(isinstance(path, str) for path in listed) or sorted(encoded) != sorted(listed):
        raise DeliveryError("delivered files differ from their manifest")
    try:
        files = {path: base64.b64decode(encoded[path], validate=True) for path in listed}
    except (binascii.Error, TypeError, ValueError) as error:
        raise DeliveryError("delivered file is not valid base64") from error
    rejected = manifest.get("rejected")
    if not isinstance(rejected, list):
        raise DeliveryError("malformed delivery manifest")
    expected, _ = manifest_for(files, rejected, track=track)
    if expected != manifest:
        raise DeliveryError("delivery manifest differs from the canonical manifest of its files")
    verify_manifest(manifest, files)
    if files["solver.py"] != solver.encode():
        raise DeliveryError("solver.py differs from the delivered target")
    return manifest, files


def _clear_sources(directory: Path) -> None:
    sources = directory / "cpp"
    if sources.is_symlink() or (sources.exists() and not sources.is_dir()):
        raise RuntimeEnvironmentError("The attempt's cpp/ is not a real directory")
    if sources.exists():
        shutil.rmtree(sources)


def _write_sources(directory: Path, files: dict[str, bytes]) -> bool:
    """Materialize cpp/ for inspection; skipped (False) when names collide on a case-insensitive host file system."""
    names = [path for path in sorted(files) if path.startswith("cpp/")]
    if len({path.casefold() for path in names}) != len(names):
        return False
    sources = directory / "cpp"
    sources.mkdir(mode=0o700)
    for path in names:
        descriptor = os.open(directory / path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(files[path])
    return True


def _atomic_json(path: Path, value) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def load_delivery(target: Path, log_dir: Path, step: str = "agent") -> tuple[dict, dict[str, bytes]]:
    """The committed delivery of a successful call, verified against its manifest and the committed solver.py.

    The canonical copy is ``{step}_delivery_files.json`` (independent of host file-name case rules); ``cpp/`` next to the target
    is an inspection copy."""
    log_dir = Path(log_dir)
    metrics = json.loads(read_regular_text(log_dir, f"{step}_metrics.json", missing_ok=True) or "{}")
    if not metrics.get("ok"):
        raise DeliveryError("the call has no committed successful result")
    record = json.loads(read_regular_text(log_dir, f"{step}_delivery_manifest.json"))
    manifest = record.get("manifest")
    if manifest is None:
        raise DeliveryError("the call committed no delivery", (record.get("failure") or {}).get("rejected", []))
    stored = json.loads(read_regular_text(log_dir, f"{step}_delivery_files.json", maximum=64 * 1024 * 1024))
    if stored.get("manifest_sha256") != manifest.get("sha256"):
        raise DeliveryError("committed files belong to another manifest")
    try:
        files = {path: base64.b64decode(text, validate=True) for path, text in stored["files"].items()}
    except (binascii.Error, TypeError, ValueError, AttributeError) as error:
        raise DeliveryError("committed delivery files are unreadable") from error
    verify_manifest(manifest, files)
    try:
        solver = read_regular(Path(target).parent, Path(target).name, maximum=MAX_FILE_BYTES)
    except UnsafeArtifact as error:
        raise DeliveryError(f"solver.py: {error}") from error
    if solver != files["solver.py"] or hashlib.sha256(solver).hexdigest() != metrics.get("artifact_sha256"):
        raise DeliveryError("the committed solver.py differs from the delivery")
    return manifest, files


class MultiFileDockerCliClient(DockerCliClient):
    """The explicit multi-file version of `DockerCliClient`; never used by the paper path."""

    def __init__(self, model, *, delivery: DeliverySpec, runtime: MultiFileDockerRuntime, **kwargs):
        if not isinstance(delivery, DeliverySpec) or delivery.mode != MULTI_FILE:
            raise ValueError("MultiFileDockerCliClient requires a multi-file DeliverySpec; single-file tasks use DockerCliClient")
        if not isinstance(runtime, MultiFileDockerRuntime):
            raise ValueError("Multi-file delivery requires the multi-file runtime entry")
        super().__init__(model, runtime=runtime, **kwargs)
        self.delivery = delivery

    def protocol(self):
        return super().protocol() | {"transport_policy": MULTIFILE_POLICY, "delivery_mode": MULTI_FILE, "delivery_track": self.delivery.track,
                                     "workspace_policy": self.workspace_policy(), "extension_files_sha256": extension_file_hashes()}

    def workspace_policy(self):
        return CPP_WORKSPACE_POLICY if self.delivery.track == "cpp_allowed" else WORKSPACE_POLICY

    def prepare(self):
        key = (self.image_id, hashlib.sha256(self.runtime.seccomp_profile.read_bytes()).hexdigest(), json.dumps([worker_file_hashes(), extension_file_hashes(), CLI_VERSIONS], sort_keys=True))
        with _PREFLIGHT_LOCK:
            if key not in _PREFLIGHTS:
                # Durable preflight evidence is recovered by state path, so the path carries the whole key.
                directory = self.evidence_root / "image-preflight-multifile" / self.image_id.split(":", 1)[1] / hashlib.sha256(json.dumps(key).encode()).hexdigest()[:24]
                result = self.runtime.run(self.image_id, ["preflight"], state_path=directory / "container.json", timeout=45, artifacts=["worker.json"])
                observed = json.loads(result.artifacts.get("worker.json", "{}"))
                expected = {"codex": CLI_VERSIONS["codex-cli"], "claude": CLI_VERSIONS["claude-code-cli"], "agy": CLI_VERSIONS["agy-cli"]}
                if (result.returncode or not observed.get("ok") or observed.get("versions") != expected or observed.get("worker_files_sha256") != worker_file_hashes()
                        or observed.get("extension_files_sha256") != extension_file_hashes()):
                    raise RuntimeEnvironmentError("Multi-file Docker isolation, CLI version or worker preflight failed before authentication")
                _PREFLIGHTS.add(key)
        volume = AUTH_VOLUMES[self.backend]
        if self.backend == "agy-cli":
            self.runtime.ensure_volume(volume, "agy-native-auth", label="io.decompbench.purpose", create=False)
            return {volume: "/auth/keyrings"}
        self.runtime.ensure_volume(volume, self.backend + "-v2-auth")
        return {volume: "/auth"}

    def edit_file(self, target, prompt, *, context=None, log_dir=None, step="agent"):
        target = Path(target).absolute()
        log_dir = Path(log_dir or target.parent).resolve()
        log_dir.mkdir(parents=True, exist_ok=True)
        if target.is_symlink() or target.name != "solver.py":
            raise ValueError("Multi-file delivery edits solver.py only")
        note = edit_instruction(target.name) + (" " + CPP_NOTE if self.delivery.track == "cpp_allowed" else "")
        effective = record_workspace_prompt(Path("/work"), target.name, self.compose_prompt(target.name, prompt, context), log_dir, step,
                                            old_absolute_note=self.backend == "agy-cli", note=note, version=self.workspace_policy())
        mounts = self.prepare()
        credential = self.credential_provider(self.backend)
        job = {"target": target.name, "initial_content": read_regular_text(target.parent, target.name, missing_ok=True) or "", "prompt": effective,
               "model": self.model, "effort": self.effort, "role": self.role, "timeout": self.timeout, "credential": credential,
               "delivery": {"mode": MULTI_FILE, "track": self.delivery.track}}
        result = self.runtime.run(self.image_id, ["agent", self.backend], payload=job, state_path=log_dir / f"{step}_container.json", timeout=self.timeout, network="bridge", volumes=mounts,
                                  artifacts=[target.name, "events.jsonl", "stderr.log", "response.md", "worker.json", "delivery.json", BUNDLE_NAME],
                                  public_metadata={"backend": self.backend, "target": target.name, "model": self.model, "role": self.role, "delivery_mode": MULTI_FILE, "delivery_track": self.delivery.track})
        return self._record(target, log_dir, step, result)

    def _record(self, target, log_dir, step, result):
        """Commit order: canonical files, inspection copy and final manifest record first, then the paper bookkeeping (solver.py,
        metrics, invocation). The paper metrics are the commit marker: `load_delivery` requires successful metrics whose
        artifact_sha256 matches solver.py, and a crash before them is recovered from the persisted container result. The
        invocation's delivery fields are informational and written after the metrics."""
        track = self.delivery.track
        manifest = files = failure = None
        try:
            manifest, files = verify_bundle(result.artifacts.get(BUNDLE_NAME, ""), result.artifacts.get(target.name, ""), track)
        except DeliveryError as error:
            failure = {"error": str(error), "rejected": error.rejected}
        artifacts = {name: value for name, value in result.artifacts.items() if name != BUNDLE_NAME}
        if target.is_symlink():
            raise ValueError("Docker edits require an ordinary declared target")
        _clear_sources(target.parent)
        files_path, record_path = log_dir / f"{step}_delivery_files.json", log_dir / f"{step}_delivery_manifest.json"
        materialized = False
        if failure is None:
            _atomic_json(files_path, {"manifest_sha256": manifest["sha256"], "files": {path: base64.b64encode(files[path]).decode("ascii") for path in sorted(files)}})
            materialized = _write_sources(target.parent, files)
            _atomic_json(record_path, {"version": RECORD_VERSION, "mode": MULTI_FILE, "track": track, "manifest": manifest, "failure": None,
                                       "manifest_sha256": manifest["sha256"], "sources_materialized": materialized})
        else:
            # An invalid delivery is an undelivered target: no solver.py content and no sources are committed.
            artifacts[target.name] = ""
            target.write_text("")
        answer = super()._record(target, log_dir, step, replace(result, artifacts=artifacts))
        if not answer.ok:
            manifest = files = None
            materialized = False
            _clear_sources(target.parent)
            files_path.unlink(missing_ok=True)
        if failure is not None and answer.terminal_state == "completed" and answer.returncode == 0 and answer.error == "Completed CLI did not deliver the requested file":
            answer = replace(answer, error="Completed CLI delivery violates the multi-file contract: " + failure["error"])
        record = {"version": RECORD_VERSION, "mode": MULTI_FILE, "track": track, "manifest": manifest, "failure": failure,
                  "manifest_sha256": manifest["sha256"] if manifest else None, "sources_materialized": materialized}
        _atomic_json(record_path, record)
        invocation_path = log_dir / f"{step}_invocation.json"
        invocation = json.loads(invocation_path.read_text())
        invocation.update(delivery_mode=MULTI_FILE, delivery_track=track, delivery_manifest_sha256=record["manifest_sha256"], delivery_failure=failure["error"] if failure else None)
        _atomic_json(invocation_path, invocation)
        save_result(log_dir, step, answer)
        return answer
