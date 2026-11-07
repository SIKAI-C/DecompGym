"""The read-only task files of a mounted-files call (prompt_protocol.context_delivery = "mounted-files-v1").

The paired runner hands the client the same context list as in the pasted protocol (the instance data.json and, per
call kind, the files of earlier attempts). Here each existing file is read safely (src.agent.safe_artifacts: no links,
regular single-link files, a size limit), mapped to its location in the agent workspace by the call's declared layout
(src.agent.specs.MountLayout; MOUNT_LAYOUT by default, which src.agent.prompts.mount_location applies) and described by
its size and sha256. DockerRuntime.run mounts the files read-only at those locations (one Docker volume per layout
directory); nothing else of the host, in particular no meta.json, gold file or sidecar, is ever placed in the workspace.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from src.agent.prompts.mounted import CONTEXT_DELIVERY_MOUNTED
from src.agent.safe_artifacts import MAX_ARTIFACT_BYTES, read_regular
from src.agent.specs import CONTEXT_DIRECTORIES, MOUNT_LAYOUT, SINGLE_FILE_DELIVERY, WORKSPACE
from src.utils.errors import ProtocolError


# A mounted output.json of an earlier attempt can be as large as the largest solver output readback limit
# (src.experiments.sandbox.OUTPUT_LIMITS); every other context file keeps the artifact limit.
MAX_OUTPUT_JSON_BYTES = 32 * 1024 * 1024


@dataclass(frozen=True)
class MountedFile:
    path: str  # absolute path inside the agent container, e.g. /work/history/attempt_01/solver.py
    source: str  # the host label of the pasted protocol: <parent directory>/<name>
    contents: bytes
    description: str
    directory: str | None = None  # the read-only mount directory holding `path` (its layout role's)

    @property
    def size(self) -> int:
        return len(self.contents)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.contents).hexdigest()


def mounted_files(context, layout=MOUNT_LAYOUT) -> list[MountedFile]:
    """The mounted files of a context list, in its order; absent files are skipped as in the pasted protocol."""
    files = []
    for path in context or []:
        path = Path(path)
        if not path.exists():
            continue
        location, description, role = layout.locate(path)
        limit = MAX_OUTPUT_JSON_BYTES if path.name == "output.json" else MAX_ARTIFACT_BYTES
        files.append(MountedFile(location, f"{path.parent.name}/{path.name}", read_regular(path.parent, path.name, maximum=limit), description, role.directory))
    locations = [item.path for item in files]
    if len(set(locations)) != len(locations):
        raise ProtocolError("Two context files map to the same mounted path")
    return files


def volume_contents(files: list[MountedFile]) -> dict[str, dict[str, bytes]]:
    """{mount directory: {relative path: contents}} for DockerRuntime.run(context=...)."""
    contents: dict[str, dict[str, bytes]] = {}
    for item in files:
        directory = item.directory or next(prefix for prefix in CONTEXT_DIRECTORIES if item.path.startswith(prefix + "/"))
        contents.setdefault(directory, {})[item.path[len(directory) + 1:]] = item.contents
    return contents


def manifest(files: list[MountedFile]) -> list[dict]:
    return [{"path": item.path, "source": item.source, "bytes": item.size, "sha256": item.sha256} for item in files]


def evidence(files: list[MountedFile], target: str, delivery=SINGLE_FILE_DELIVERY, layout=MOUNT_LAYOUT) -> dict:
    """The call evidence: the layout, every mounted file's sha256 and the writable target (a multi-file delivery and a
    non-default layout also record their ids)."""
    entries = manifest(files)
    value = {"policy": CONTEXT_DELIVERY_MOUNTED, "workspace": WORKSPACE, "writable_target": f"{WORKSPACE}/{target}", "read_only_mounts": sorted(volume_contents(files)),
             "files": entries, "manifest_sha256": hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest()}
    if delivery.sources(target):
        value["writable"] = [f"{WORKSPACE}/{target}", f"{WORKSPACE}/cpp/*.cpp", f"{WORKSPACE}/cpp/*.hpp"]
        value["delivery"] = delivery.id
    if layout != MOUNT_LAYOUT:
        value["layout"] = layout.id
    return value
