"""Docker lifecycle with durable handles and regular-file collection after exit."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import io
import json
import os
import re
from pathlib import Path, PurePosixPath
import selectors
import subprocess
import tarfile
import threading
import time
import uuid

from src.agent.prompts.mounted import CONTEXT_DIRECTORIES
from src.experiments.resources import CleanupError
from src.utils.errors import InfrastructureError
from src.experiments.environment import RuntimeEnvironmentError


class ContainerError(InfrastructureError):
    """Container infrastructure is unavailable; candidate performance is unknown."""


@dataclass
class ContainerResult:
    returncode: int
    stdout: str
    stderr: str
    elapsed_seconds: float
    timed_out: bool = False
    output_limited: bool = False
    oom_killed: bool = False
    artifacts: dict[str, str] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)


def bounded_process(command: list[str], *, stdin: str = "", timeout: float, max_output_bytes: int = 8 * 1024 * 1024, heartbeat: bool = False) -> ContainerResult:
    """Drain both pipes with a fixed total ceiling and a monotonic deadline."""
    started = time.monotonic()
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    input_error = []
    finished = threading.Event()

    def send_input():
        try:
            process.stdin.write(stdin.encode())
            process.stdin.flush()
            while heartbeat and not finished.wait(1):
                process.stdin.write(b"heartbeat\n")
                process.stdin.flush()
            process.stdin.close()
        except (BrokenPipeError, OSError, ValueError) as error:
            input_error.append(type(error).__name__)

    sender = threading.Thread(target=send_input, daemon=True)
    sender.start()
    streams = selectors.DefaultSelector()
    streams.register(process.stdout, selectors.EVENT_READ, "stdout")
    streams.register(process.stderr, selectors.EVENT_READ, "stderr")
    output = {"stdout": bytearray(), "stderr": bytearray()}
    timed_out = limited = False
    try:
        while streams.get_map():
            remaining = timeout - (time.monotonic() - started)
            if remaining <= 0:
                timed_out = True
                break
            for key, _ in streams.select(min(0.2, remaining)):
                chunk = os.read(key.fileobj.fileno(), 65536)
                if not chunk:
                    streams.unregister(key.fileobj)
                    continue
                used = sum(len(value) for value in output.values())
                output[key.data].extend(chunk[:max(0, max_output_bytes - used)])
                if used + len(chunk) > max_output_bytes:
                    limited = True
                    break
            if limited:
                break
        if timed_out or limited:
            process.kill()
        try:
            process.wait(timeout=max(0.2, timeout - (time.monotonic() - started)))
        except subprocess.TimeoutExpired:
            timed_out = True
            process.kill()
            process.wait(timeout=5)
    finally:
        finished.set()
        streams.close()
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        sender.join(timeout=1)
        for stream in (process.stdout, process.stderr):
            stream.close()
    return ContainerResult(process.returncode, output["stdout"].decode(errors="replace"), output["stderr"].decode(errors="replace"), time.monotonic() - started, timed_out, limited)


def context_archive(files: dict[str, bytes]) -> bytes:
    """A deterministic tar of read-only task files (mode 0444) and their directories (0555) for `docker cp -`."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.PAX_FORMAT) as archive:
        for directory in sorted({str(parent) for name in files for parent in PurePosixPath(name).parents if str(parent) != "."}):
            info = tarfile.TarInfo(directory)
            info.type, info.mode, info.mtime = tarfile.DIRTYPE, 0o555, 0
            archive.addfile(info)
        for name in sorted(files):
            info = tarfile.TarInfo(name)
            info.size, info.mode, info.mtime = len(files[name]), 0o444, 0
            archive.addfile(info, io.BytesIO(files[name]))
    return buffer.getvalue()


def archive_manifest(data: bytes, root: str) -> dict[str, str]:
    """{relative path: sha256} of the regular files of a `docker cp <container>:<root> -` tar; any other entry type is refused."""
    found = {}
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:") as archive:
        for member in archive.getmembers():
            parts = PurePosixPath(member.name).parts
            if parts and parts[0] == PurePosixPath(root).name:
                parts = parts[1:]
            if member.isdir():
                continue
            if not member.isfile() or not parts or any(part in ("..", "/") for part in parts):
                raise RuntimeEnvironmentError("Mounted task context holds an unexpected entry")
            found[str(PurePosixPath(*parts))] = hashlib.sha256(archive.extractfile(member).read()).hexdigest()
    return found


def check_context(context) -> None:
    """Context mounts: {directory in CONTEXT_DIRECTORIES: {normalized relative path: bytes}}, every directory non-empty."""
    if not isinstance(context, dict) or not set(context) <= set(CONTEXT_DIRECTORIES) or not all(isinstance(files, dict) and files for files in context.values()):
        raise ValueError("Context mounts must map declared read-only directories to files")
    for files in context.values():
        for name, contents in files.items():
            path = PurePosixPath(name)
            if not isinstance(contents, bytes) or path.is_absolute() or str(path) != name or any(part in (".", "..") for part in path.parts):
                raise ValueError("Context files need normalized relative paths and bytes")


def save_state(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


class DockerRuntime:
    LABEL = "io.decompbench.runtime"
    CONTEXT_LOADER = "v2-context-loader"  # label value of the never-started container that fills read-only context volumes
    # The image's trusted worker and the collector's output ceiling. The paper path uses these defaults; only the
    # multi-file runtime (src/agent/multifile.py, PDPTW C++ track) overrides them in a subclass.
    WORKER = "/opt/decompbench/worker.py"
    COLLECT_OUTPUT_BYTES = 8 * 1024 * 1024

    def __init__(self, *, executable="docker", seccomp_profile: Path):
        self.executable = executable
        self.seccomp_profile = seccomp_profile.resolve()
        self.expected_engine = None

    def cli(self, *args):
        connection = ["--host", self.expected_engine["docker_endpoint"]] if self.expected_engine else []
        return [self.executable, *connection, *args]

    def bind_environment(self, value):
        if not value.get("docker_endpoint", "").startswith("unix://"):
            raise RuntimeEnvironmentError("The reviewed workflow requires the declared local Docker endpoint")
        self.expected_engine = value
        self.validate_engine()

    def validate_engine(self):
        if self.expected_engine is None:
            return
        identity = self.command("info", "--format", "{{.ID}}")
        version = self.command("version", "--format", "{{.Client.Version}} {{.Server.Version}}")
        if identity.returncode or version.returncode:
            raise ContainerError("The declared Docker engine is unavailable")
        expected_version = self.expected_engine["docker_client_version"] + " " + self.expected_engine["docker_server_version"]
        if identity.stdout.strip() != self.expected_engine["docker_engine_id"] or version.stdout.strip() != expected_version:
            raise RuntimeEnvironmentError("Docker engine identity or version changed from the frozen environment")

    def resource_samples(self):
        listed = self.command("ps", "--filter", f"label={self.LABEL}=v2-worker", "--format", "{{.Names}}")
        if listed.returncode:
            raise ContainerError("Cannot monitor Docker worker resources")
        names = listed.stdout.split()
        if not names:
            return []
        stats = self.command("stats", "--no-stream", "--format", "{{json .}}", *names)
        if not stats.returncode:
            return [parse_docker_sample(json.loads(line)) for line in stats.stdout.splitlines() if line.strip()]
        # A worker may exit between `ps` and `stats`: sample the workers one by one and skip those that are gone.
        samples = []
        for name in names:
            for _ in range(2):
                single = self.command("stats", "--no-stream", "--format", "{{json .}}", name)
                if not single.returncode:
                    samples += [parse_docker_sample(json.loads(line)) for line in single.stdout.splitlines() if line.strip()]
                    break
                observed = self.inspect(name)
                if observed is None or not observed["State"]["Running"]:
                    break
            else:
                raise ContainerError("Cannot sample Docker worker resources")
        return samples

    def command(self, *args, timeout=30):
        try:
            return subprocess.run(self.cli(*args), capture_output=True, text=True, timeout=timeout)
        except (OSError, subprocess.TimeoutExpired) as error:
            raise ContainerError("Docker command could not complete") from error

    def image_id(self, image):
        result = self.command("image", "inspect", image, "--format", "{{.Id}}")
        if result.returncode and "no such image" in result.stderr.lower():
            raise RuntimeEnvironmentError(f"Required frozen Docker image is unavailable: {image}")
        if result.returncode or not result.stdout.strip().startswith("sha256:"):
            raise ContainerError(f"Required cached image is unavailable: {image}")
        return result.stdout.strip()

    def transfer(self, *args, data=None, timeout=120):
        """A Docker CLI call with binary standard input and output (`docker cp` archives)."""
        try:
            return subprocess.run(self.cli(*args), input=b"" if data is None else data, capture_output=True, timeout=timeout)
        except (OSError, subprocess.TimeoutExpired) as error:
            raise ContainerError("Docker context transfer could not complete") from error

    def ensure_volume(self, name, purpose, *, label=None, create=True):
        label = label or self.LABEL
        result = self.command("volume", "inspect", name, "--format", "{{json .Labels}}")
        if result.returncode:
            if not create or "no such volume" not in result.stderr.lower():
                raise ContainerError("The dedicated Docker volume is unavailable")
            created = self.command("volume", "create", "--label", f"{label}={purpose}", name)
            if created.returncode:
                raise ContainerError("Could not create the dedicated Docker volume")
            result = self.command("volume", "inspect", name, "--format", "{{json .Labels}}")
        if result.returncode or (json.loads(result.stdout or "null") or {}).get(label) != purpose:
            raise RuntimeEnvironmentError("Docker volume ownership label does not match")

    def volume_users(self, volume):
        """(name, io.decompbench.runtime label) of each running container that mounts `volume`."""
        result = self.command("ps", "--filter", f"volume={volume}", "--filter", "status=running", "--format", "{{.Names}}\t{{.Label \"" + self.LABEL + "\"}}")
        if result.returncode:
            raise ContainerError("Cannot list the containers that use an auth volume")
        return [tuple((line.split("\t") + [""])[:2]) for line in result.stdout.splitlines() if line.strip()]

    def inspect(self, name):
        result = self.command("container", "inspect", name)
        if result.returncode:
            if "no such container" in result.stderr.lower() or "no such object" in result.stderr.lower():
                return None
            raise ContainerError("Docker container state is temporarily unavailable")
        return json.loads(result.stdout)[0]

    def remove_container(self, name):
        self.command("rm", "--force", name)
        if self.inspect(name) is not None:
            raise CleanupError(f"Could not verify container cleanup: {name}")

    def stop_campaign_workers(self, root: Path):
        """After the host owner exits, verify only this campaign's durable handles."""
        result = self.command("ps", "--filter", f"label={self.LABEL}=v2-worker", "--format", "{{.Names}}")
        if result.returncode:
            raise CleanupError("Cannot verify Docker worker state after the controller exited")
        running = set(result.stdout.split())
        if not running:
            return
        owned = {}
        for path in root.rglob("*container.json"):
            state = json.loads(path.read_text())
            if state.get("name") in running:
                owned[state["name"]] = state
        deadline = time.monotonic() + 8
        while owned:
            for name, state in list(owned.items()):
                observed = self.inspect(name)
                if observed is None or not observed.get("State", {}).get("Running"):
                    owned.pop(name)
                    continue
                labels = observed.get("Config", {}).get("Labels") or {}
                volumes = {item.get("Name") for item in observed.get("Mounts", []) if item.get("Type") == "volume"}
                if labels.get(self.LABEL) != "v2-worker" or not {state["work_volume"], state["record_volume"]} <= volumes:
                    raise CleanupError("Docker worker ownership differs from the campaign receipt")
                if time.monotonic() >= deadline:
                    # Preserve both volumes so recovery can collect already produced artifacts.
                    self.remove_container(name)
                    owned.pop(name)
            if owned:
                time.sleep(0.25)

    def _dispose(self, state):
        self.remove_container(state["name"])
        if "context_volumes" in state:
            self.remove_container(state["name"] + "-context")
        for volume in (state["work_volume"], state["record_volume"], *state.get("context_volumes", {}).values()):
            result = self.command("volume", "rm", volume)
            if result.returncode and "no such volume" not in result.stderr.lower():
                raise CleanupError(f"Could not remove owned temporary volume: {volume}")

    def run(self, image, command, *, payload=None, state_path: Path, timeout=900, network="none", cpus=1, memory="1024m", volumes=None, artifacts=None, public_metadata=None, context=None):
        """`context` (mounted-files protocol only): {directory: {relative path: bytes}}, mounted read-only at each
        directory of src.agent.prompts.mounted.CONTEXT_DIRECTORIES through a fresh volume; None keeps the paper run."""
        self.validate_engine()
        if network not in {"none", "bridge"} or timeout <= 0 or cpus <= 0:
            raise ValueError("Invalid Docker execution settings")
        if any(Path(name).name != name or name in ("", ".", "..") for name in artifacts or []):
            raise ValueError("Artifact names must be basenames")
        if context is not None:
            check_context(context)
        if state_path.exists():
            return self.recover(state_path)
        identifier = uuid.uuid4().hex
        state = {"schema_version": 1, "name": f"decompbench-v2-{identifier}", "work_volume": f"decompbench-v2-work-{identifier}", "record_volume": f"decompbench-v2-record-{identifier}",
                 "image_id": self.image_id(image), "command": command, "artifacts": artifacts or [], "created_at": time.time(), "timeout": timeout, "network": network, "cpus": cpus, "memory": memory,
                 "phase": "preparing", "public_metadata": public_metadata or {}, "credential_values_recorded": False}
        if context:
            state["context_volumes"] = {destination: f"decompbench-v2-context-{PurePosixPath(destination).name}-{identifier}" for destination in sorted(context)}
            state["context_manifest"] = {destination: {name: hashlib.sha256(data).hexdigest() for name, data in sorted(context[destination].items())} for destination in sorted(context)}
        save_state(state_path, state)
        try:
            for volume in (state["work_volume"], state["record_volume"]):
                self.ensure_volume(volume, "v2-ephemeral-worker")
            if context:
                self._load_context(state, context)
            args = ["create", "--name", state["name"], "--label", f"{self.LABEL}=v2-worker", "--pull", "never", "--platform", "linux/arm64", "--init", "--user", "1000:1000", "--cap-drop", "ALL",
                    "--security-opt", "no-new-privileges", "--security-opt", f"seccomp={self.seccomp_profile}", "--read-only", "--log-driver", "none", "--pids-limit", "128", "--memory", memory, "--memory-swap", memory,
                    "--cpus", str(cpus), "--network", network, "--tmpfs", "/tmp:rw,nosuid,nodev,size=128m", "--tmpfs", "/home/node:rw,nosuid,nodev,size=128m,uid=1000,gid=1000,mode=0700",
                    "--tmpfs", "/run/user/1000:rw,nosuid,nodev,size=16m,uid=1000,gid=1000,mode=0700", "--mount", f"type=volume,src={state['work_volume']},dst=/work", "--mount", f"type=volume,src={state['record_volume']},dst=/record", "--interactive", "--entrypoint", "python3"]
            for volume, destination in (volumes or {}).items():
                if "/" in volume or not (destination == "/auth" or destination.startswith("/auth/")):
                    raise ValueError("Credentials must use a named volume under /auth")
                args.extend(["--mount", f"type=volume,src={volume},dst={destination}"])
            for destination, volume in state.get("context_volumes", {}).items():
                args.extend(["--mount", f"type=volume,src={volume},dst={destination},readonly"])
            created = self.command(*args, state["image_id"], "-I", "-B", self.WORKER, *command)
            if created.returncode:
                raise ContainerError("Docker worker creation failed")
            inspected = self.inspect(state["name"])
            configuration = inspected["HostConfig"]
            mounts = [{key: mount.get(key) for key in ("Type", "Name", "Destination", "RW")} for mount in inspected["Mounts"]]
            if configuration.get("Privileged") or not configuration.get("ReadonlyRootfs") or configuration.get("Binds") or any(mount["Type"] == "bind" for mount in mounts):
                raise RuntimeEnvironmentError("Docker worker isolation differs from the declared configuration")
            for destination, volume in state.get("context_volumes", {}).items():
                if {"Type": "volume", "Name": volume, "Destination": destination, "RW": False} not in mounts:
                    raise RuntimeEnvironmentError("Mounted task context is not read-only as declared")
            state.update(phase="starting", container_id=created.stdout.strip(), mounts=mounts, security_options=configuration["SecurityOpt"], deadline=time.time() + timeout)
            save_state(state_path, state)
            attached = bounded_process(self.cli("start", "--attach", "--interactive", state["name"]), stdin=json.dumps(payload or {}) + "\n", timeout=timeout, heartbeat=True)
            return self._finish(state_path, state, attached)
        except BaseException:
            # Keep stopped volumes until their logs and targets can be collected on recovery.
            self.remove_container(state["name"])
            if state["phase"] == "preparing":
                self._dispose(state)
            elif not state_path.with_name(state_path.stem + "_result.json").exists():
                state.setdefault("transport_result", asdict(ContainerResult(1, "", "Controller interrupted", 0, metadata={"usage_unavailable": True})))
                state["phase"] = "stopped"
                save_state(state_path, state)
            raise

    def _load_context(self, state, context):
        """Fill each context volume through a never-started container (`docker cp -`) and read it back against the manifest."""
        loader = state["name"] + "-context"
        for volume in state["context_volumes"].values():
            self.ensure_volume(volume, "v2-ephemeral-context")
        args = ["create", "--name", loader, "--label", f"{self.LABEL}={self.CONTEXT_LOADER}", "--pull", "never", "--platform", "linux/arm64", "--network", "none", "--cap-drop", "ALL",
                "--security-opt", "no-new-privileges", "--log-driver", "none"]
        for destination, volume in state["context_volumes"].items():
            args.extend(["--mount", f"type=volume,src={volume},dst={destination}"])
        if self.command(*args, "--entrypoint", "python3", state["image_id"], "-I", "-B", "-c", "").returncode:
            raise ContainerError("Docker context loader creation failed")
        try:
            for destination in sorted(context):
                if self.transfer("cp", "-", f"{loader}:{destination}", data=context_archive(context[destination])).returncode:
                    raise ContainerError("Copying the task context into its volume failed")
                copied = self.transfer("cp", f"{loader}:{destination}", "-")
                if copied.returncode or archive_manifest(copied.stdout, destination) != state["context_manifest"][destination]:
                    raise RuntimeEnvironmentError("Mounted task context differs from its manifest")
        finally:
            self.remove_container(loader)

    def recover(self, state_path):
        self.validate_engine()
        state = json.loads(state_path.read_text())
        result_path = state_path.with_name(state_path.stem + "_result.json")
        if result_path.exists():
            self._dispose(state)
            return ContainerResult(**json.loads(result_path.read_text()))
        if state["phase"] == "stopped":
            return self._finish(state_path, state, ContainerResult(**state["transport_result"]))
        observed = self.inspect(state["name"])
        if observed is None:
            if state["phase"] != "preparing":
                return self._finish(state_path, state, ContainerResult(1, "", "Execution container is absent; recovering its volumes", 0, metadata={"recovered_existing_container": True}))
            self._dispose(state)
            result = ContainerResult(1, "", "Interrupted container is no longer available", 0, metadata={"recovery": "missing_container", "usage_unavailable": True})
            save_state(result_path, asdict(result))
            return result
        if observed["State"]["Status"] == "created":
            self._dispose(state)
            result = ContainerResult(1, "", "Interrupted before the worker received its request", 0, metadata={"recovery": "not_started", "usage_unavailable": True})
            save_state(result_path, asdict(result))
            return result
        # Poll the existing handle; no inference request is resent on recovery.
        remaining = max(0.1, state["deadline"] - time.time())
        waited = bounded_process(self.cli("wait", state["name"]), timeout=remaining) if observed["State"]["Running"] else ContainerResult(observed["State"]["ExitCode"], "", "", 0)
        waited.metadata["recovered_existing_container"] = True
        return self._finish(state_path, state, waited)

    def _finish(self, state_path, state, result):
        observed = self.inspect(state["name"])
        if observed is not None:
            result.oom_killed = bool(observed["State"].get("OOMKilled"))
            if not observed["State"]["Running"]:
                result.returncode = observed["State"]["ExitCode"]
        self.remove_container(state["name"])
        state["phase"] = "stopped"
        state["transport_result"] = asdict(result)
        save_state(state_path, state)
        # The execution container has gone, so no task process can mutate these volumes.
        collector = state["name"] + "-collect"
        args = [*self.cli("run"), "--rm", "--name", collector, "--pull", "never", "--network", "none", "--user", "1000:1000", "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--read-only", "--log-driver", "none", "--pids-limit", "32", "--memory", "256m", "--cpus", "1",
                "--mount", f"type=volume,src={state['work_volume']},dst=/work,readonly", "--mount", f"type=volume,src={state['record_volume']},dst=/record,readonly", "--entrypoint", "python3", state["image_id"], "-I", "-B", self.WORKER, "collect", *state["artifacts"]]
        try:
            collected = bounded_process(args, timeout=30) if self.COLLECT_OUTPUT_BYTES == 8 * 1024 * 1024 else bounded_process(args, timeout=30, max_output_bytes=self.COLLECT_OUTPUT_BYTES)
            if collected.returncode or collected.timed_out or collected.output_limited:
                raise ContainerError("Artifact collector failed after worker cleanup")
            packet = json.loads(collected.stdout)
            if not packet.get("ok"):
                raise ContainerError("Trusted container records failed collection checks")
            result.artifacts = packet["artifacts"]
            result.metadata.update(state | {"cleanup_verified": True, "rejected_artifacts": packet.get("rejected", [])})
            result.elapsed_seconds = time.time() - state["created_at"]
            save_state(state_path.with_name(state_path.stem + "_result.json"), asdict(result))
        finally:
            self.remove_container(collector)
        self._dispose(state)
        state["phase"] = "collected"
        save_state(state_path, state)
        return result


def parse_docker_sample(value):
    memory = value["MemUsage"].split("/", 1)[0].strip()
    match = re.fullmatch(r"([0-9.]+)\s*([kKMGT]?i?B)", memory)
    if not match:
        raise ContainerError("Unrecognized Docker memory usage units")
    factors = {"B": 1 / 2**20, "kB": 1000 / 2**20, "KB": 1000 / 2**20, "KiB": 1 / 1024, "MB": 1000**2 / 2**20, "MiB": 1, "GB": 1000**3 / 2**20, "GiB": 1024, "TB": 1000**4 / 2**20, "TiB": 1024**2}
    return {"name": value.get("Name", value.get("Container")), "cpu_percent": float(value["CPUPerc"].rstrip("%")), "rss_mib": float(match[1]) * factors[match[2]]}
