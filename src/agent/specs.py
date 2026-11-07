"""The agent-side specifications of a call: where the read-only context is placed, what the agent may write, and the
run-time contract it is told.

Three independent, versioned parts (the runner-side choice of which files a call receives is
src.experiments.context_spec.ContextSpec, which carries one MountLayout):

- MountLayout (`id`): the declared roles of read-only context files. A role admits host files by name, at the case level
  or inside an attempt_NN directory, and mounts them in one read-only directory of the agent workspace. Any other file
  is refused (fail-closed). MOUNT_LAYOUT (instance data.json + history files) is the layout of mounted-files-v1;
  PREVIEW_LAYOUT adds the structural preview of data.json and is not used by any config.
- DeliverySpec (`id`): the writable and collected targets of a generation call. SINGLE_FILE_DELIVERY (solver.py, and
  debug.md for a reflection) is the paper's and every config's; the multi-file mode (solver.py plus flat cpp/ sources,
  src.agent.multifile) is the large-scale PDPTW C++ track. Both compose with pasted and mounted context.
- RuntimeSpec (`id`): the run-time contract (input and output paths, time and thread budget, output readback policy,
  optional publication API). RuntimeSpec itself is the paired protocol's separate-edit-runtime-v1 (the
  prompt_protocol.runtime_location id): it renders exactly the earlier fixed sentences. Another runtime (for example a
  large-scale supervisor run with cwd /work) is a subclass that overrides the class constants and the rendering methods.

None of these ids is written into the records of an earlier protocol: they are recorded only when a runner uses a
non-default spec (src.experiments.paired_runner.PairedRunner.spec_record).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fnmatch import fnmatchcase
from pathlib import Path, PurePosixPath
import re
from types import MappingProxyType
from typing import ClassVar, Mapping

from src.utils.errors import ProtocolError


# ------------------------------------------------------------------------------------------------ read-only context

WORKSPACE = "/work"
INSTANCE_DIR = WORKSPACE + "/instance"
HISTORY_DIR = WORKSPACE + "/history"
REVIEW_DIR = WORKSPACE + "/review"  # the method judge's sources under review (REVIEW_LAYOUT)
# The read-only directories of MOUNT_LAYOUT (DockerRuntime.run accepts these unless a call declares others).
CONTEXT_DIRECTORIES = (INSTANCE_DIR, HISTORY_DIR)
ATTEMPT_DIRECTORY = re.compile(r"attempt_(\d{2})")
INSTANCE_FILE = "the complete instance data (data.json) of this task"
HISTORY_FILES = {
    "solver.py": "the solver.py of attempt {n}",
    "debug.md": "the reflection notes (debug.md) written after attempt {n}",
    "reflection_feedback.md": "framework feedback from the harness (not a model reflection) about the reflection call after attempt {n}",
    "exec.log": "the harness log of attempt {n}: solver execution output, verification issues and the algorithm review",
    "alg_verifier.md": "the method judge's review (an LLM-based algorithm check) of the solver.py of attempt {n}",
    "output.json": "the output.json written by the solver.py of attempt {n}",
}
# The structural preview of a large data.json (src.experiments.data_context), written next to the case's data.json:
# PREVIEW_NAME replaces data.json (full-or-structural-preview-512KiB-v1), PREVIEW_ALONGSIDE_NAME accompanies it.
PREVIEW_NAME = "data.json.structural-preview.txt"
PREVIEW_ALONGSIDE_NAME = "data.json.structural-preview-alongside.txt"
PREVIEW_FILE = "a structural preview of data.json (the first items of every list and each list's true length), not the data itself"
PREVIEW_ALONGSIDE_FILE = "a structural preview of the mounted data.json (the first items of every list and each list's true length)"
# A mount directory: one plain name below the workspace.
MOUNT_DIRECTORY = re.compile(re.escape(WORKSPACE) + r"/[A-Za-z0-9_][A-Za-z0-9_.-]*")
# Part 1 sections a role's files are listed under.
INPUT_SECTION, HISTORY_SECTION, REVIEW_SECTION = "Input data", "Files from previous attempts", "Files under review"


@dataclass(frozen=True)
class ContextRole:
    """One declared kind of read-only context file.

    `files` maps an admitted name to its description. A case-level role admits files by name next to the case's data
    (never directly inside an attempt directory). A per-attempt role admits attempt-relative paths inside attempt_NN/,
    given exactly or as a glob pattern (for example cpp/*.cpp), and mounts them at <directory>/attempt_NN/<path>;
    descriptions may use {n} (the attempt number) and {path} (the attempt-relative path)."""
    name: str
    directory: str  # read-only mount directory inside the workspace
    files: Mapping[str, str] = field(hash=False)
    per_attempt: bool = False
    section: str = INPUT_SECTION

    def __post_init__(self):
        object.__setattr__(self, "files", MappingProxyType(dict(self.files)))  # a frozen copy: a caller's dict cannot change the role
        for name in self.files:
            parts = PurePosixPath(name).parts if isinstance(name, str) else ()
            if not parts or PurePosixPath(name).is_absolute() or any(part in (".", "..") for part in parts) or (not self.per_attempt and len(parts) != 1):
                raise ValueError(f"A context role admits normalized relative names: {name!r}")

    def locate(self, path: Path) -> tuple[str, str] | None:
        """(mounted path, description) of a host file this role admits, else None."""
        if not self.per_attempt:
            if path.name not in self.files or ATTEMPT_DIRECTORY.fullmatch(path.parent.name):
                return None
            return f"{self.directory}/{path.name}", self.files[path.name]
        attempt = next((parent for parent in path.parents if ATTEMPT_DIRECTORY.fullmatch(parent.name)), None)
        if attempt is None:
            return None
        relative = path.relative_to(attempt).as_posix()
        key = relative if relative in self.files else next((name for name in self.files if any(c in name for c in "*?[") and fnmatchcase(relative, name)), None)
        if key is None:
            return None
        return f"{self.directory}/{attempt.name}/{relative}", self.files[key].format(n=int(ATTEMPT_DIRECTORY.fullmatch(attempt.name)[1]), path=relative)


INSTANCE_ROLE = ContextRole("instance_data", INSTANCE_DIR, {"data.json": INSTANCE_FILE})
HISTORY_ROLE = ContextRole("history", HISTORY_DIR, HISTORY_FILES, per_attempt=True, section=HISTORY_SECTION)
PREVIEW_ROLE = ContextRole("data_preview", INSTANCE_DIR, {PREVIEW_NAME: PREVIEW_FILE, PREVIEW_ALONGSIDE_NAME: PREVIEW_ALONGSIDE_FILE})
# The C++ sources of earlier attempts of a multi-file delivery (no config uses them).
SOURCE_HISTORY_ROLE = ContextRole("history_sources", HISTORY_DIR, {"cpp/*.cpp": "the C++ source {path} of attempt {n}", "cpp/*.hpp": "the C++ header {path} of attempt {n}"},
                                  per_attempt=True, section=HISTORY_SECTION)


@dataclass(frozen=True)
class MountLayout:
    """The roles a mounted call may use, tried in order; a file no role admits is refused."""
    id: str
    roles: tuple[ContextRole, ...]

    def __post_init__(self):
        sections = {}
        for role in self.roles:
            if not MOUNT_DIRECTORY.fullmatch(role.directory) or role.section not in (INPUT_SECTION, HISTORY_SECTION, REVIEW_SECTION):
                raise ValueError(f"A context role mounts one directory /work/<name> listed under a known Part 1 section: {role}")
            if sections.setdefault(role.directory, role.section) != role.section:
                raise ValueError(f"Roles sharing {role.directory} must be listed in the same Part 1 section")

    @property
    def directories(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(role.directory for role in self.roles))

    def locate(self, path: Path) -> tuple[str, str, ContextRole]:
        """(mounted path, description, role) of one context file of the paired runner."""
        path = Path(path)
        for role in self.roles:
            found = role.locate(path)
            if found:
                return (*found, role)
        raise ProtocolError(f"No mounted location for context file {path.parent.name}/{path.name}")

    def role_at(self, location: str) -> ContextRole:
        """The role whose directory holds a mounted path."""
        for role in self.roles:
            if location.startswith(role.directory + "/"):
                return role
        raise ProtocolError(f"Mounted path {location} is outside the declared layout")

    def section(self, location: str) -> str:
        return self.role_at(location).section

    def directory(self, location: str) -> str:
        return self.role_at(location).directory


# The method judge's sources under review (judge_context_delivery = mounted-review-sources-v1): the delivered solver.py and,
# for a multi-file delivery, its flat cpp/ sources, at /work/review/attempt_NN/<path> (the pasted label attempt_NN/<path>).
REVIEW_ROLE = ContextRole("sources_under_review", REVIEW_DIR, {"solver.py": "the solver.py delivered by attempt {n} (under review)", "cpp/*.cpp": "the C++ source {path} delivered by attempt {n} (under review)",
                                                                "cpp/*.hpp": "the C++ header {path} delivered by attempt {n} (under review)"}, per_attempt=True, section=REVIEW_SECTION)


MOUNT_LAYOUT = MountLayout("instance-and-history-v1", (HISTORY_ROLE, INSTANCE_ROLE))
PREVIEW_LAYOUT = MountLayout("instance-preview-and-history-v1", (HISTORY_ROLE, INSTANCE_ROLE, PREVIEW_ROLE))
SOURCES_LAYOUT = MountLayout("instance-and-history-with-sources-v1", (HISTORY_ROLE, SOURCE_HISTORY_ROLE, INSTANCE_ROLE))
REVIEW_LAYOUT = MountLayout("review-sources-v1", (REVIEW_ROLE,))


# ------------------------------------------------------------------------------------------------ writable delivery

SINGLE_FILE, MULTI_FILE = "single_file", "multi_file"
CPP_NOTE = ("You may also create C++17 source files as flat cpp/*.cpp and cpp/*.hpp files under the current working directory "
            "(at most 64 C++ files, 2 MiB per file, 8 MiB in total including solver.py); create no other files or directories.")


@dataclass(frozen=True)
class DeliverySpec:
    """The writable and collected files of a generation call (a reflection or judge call writes its one note)."""
    mode: str = SINGLE_FILE
    track: str | None = None  # multi-file: src.experiments.xl_build.TRACKS

    def __post_init__(self):
        from src.experiments.xl_build import TRACKS

        if self.mode == SINGLE_FILE and self.track is not None:
            raise ValueError("single-file delivery has no track")
        if self.mode == MULTI_FILE and self.track not in TRACKS:
            raise ValueError(f"multi-file delivery requires a track in {TRACKS}")
        if self.mode not in (SINGLE_FILE, MULTI_FILE):
            raise ValueError(f"unknown delivery mode {self.mode!r}")

    @property
    def id(self) -> str:
        return "single-file-target-v1" if self.mode == SINGLE_FILE else f"multi-file-{self.track}-v1"

    def sources(self, target: str) -> bool:
        """Whether a call editing `target` may also write C++ sources (the cpp_allowed track's generation)."""
        return self.mode == MULTI_FILE and self.track == "cpp_allowed" and target == "solver.py"

    @property
    def nested(self) -> tuple[str, ...]:
        """Subdirectories of an attempt that belong to its delivered artifact set (shared-first copies include them)."""
        return ("cpp",) if self.mode == MULTI_FILE and self.track == "cpp_allowed" else ()

    def reviewed_sources(self, attempt_dir: Path, call_dir: Path | None = None) -> list[Path]:
        """The delivered source files the method judge reviews, solver.py first. For a multi-file delivery these are the
        committed, verified delivery of the generation call in `call_dir` (src.agent.multifile.load_delivery); the
        attempt's cpp/ inspection copy must hold exactly those files and bytes, otherwise the review is refused."""
        attempt_dir = Path(attempt_dir)
        if not self.nested:
            return [attempt_dir / "solver.py"]
        from src.agent.multifile import load_delivery

        if call_dir is None:
            raise ProtocolError("A multi-file review needs the committed generation call")
        _, files = load_delivery(attempt_dir / "solver.py", call_dir)
        sources = sorted(path for path in files if path != "solver.py")
        copy = attempt_dir / "cpp"
        present = sorted(path.relative_to(attempt_dir).as_posix() for path in copy.rglob("*")) if copy.is_dir() and not copy.is_symlink() else []
        if present != sources or any((attempt_dir / path).is_symlink() or (attempt_dir / path).read_bytes() != files[path] for path in sources):
            raise ProtocolError("The attempt's cpp/ copy differs from the committed delivery; the method judge cannot review it")
        return [attempt_dir / "solver.py", *(attempt_dir / path for path in sources)]

    def instruction(self, target: str) -> str:
        """The edit instruction that opens the prompt (record_workspace_prompt `note`, after Part 1's headings when mounted)."""
        from src.agent.prompts.policy import edit_instruction

        return edit_instruction(target) + (" " + CPP_NOTE if self.sources(target) else "")

    def modify_text(self, target: str, workspace: str = WORKSPACE) -> str:
        """Part 1's "File to modify" rule after the legacy "Edit only <target>." prefix."""
        writable = f"{workspace}/{target}"
        if not self.sources(target):
            return (f"That file is {writable}; do not create, rename or delete any file. Every other path in this workspace is read-only context. "
                    f"Only {writable} is collected after this call: anything written elsewhere is ignored.")
        return (f"Apart from {writable} and those C++ sources in {workspace}/cpp/, every path in this workspace is read-only context. "
                "Only these files are collected after this call: anything written elsewhere is ignored.")

    def writable_text(self, target: str, workspace: str = WORKSPACE) -> str:
        """The writable files as named in Part 1's "No other task files" sentence."""
        writable = f"{workspace}/{target}"
        return f"{writable} and {workspace}/cpp/" if self.sources(target) else writable


SINGLE_FILE_DELIVERY = DeliverySpec()


# ------------------------------------------------------------------------------------------------ run-time contract

RUNTIME_LOCATION = "separate-edit-runtime-v1"  # src.agent.prompts.snapshot.PROMPT_PROTOCOL["runtime_location"]


@dataclass(frozen=True)
class RuntimeSpec:
    """The paired protocol's run-time contract: solver.py runs later in its own attempt directory (its cwd), reads
    ../../data.json and writes output.json there, within `timeout` seconds and `threads` solver threads; the harness
    reads output.json under `output_policy` (src.experiments.sandbox.OUTPUT_LIMITS: None = 4 MiB). No publication API."""
    timeout: int = 60
    threads: int = 1
    output_policy: str | None = None
    id: ClassVar[str] = RUNTIME_LOCATION
    input_path: ClassVar[str] = "../../data.json"  # relative to the run's working directory
    output_name: ClassVar[str] = "output.json"  # in the run's working directory
    publication: ClassVar[str | None] = None  # agent-facing description of a publication API (None: the final output only)

    def __post_init__(self):
        from src.experiments.sandbox import OUTPUT_LIMITS

        if self.timeout <= 0 or self.threads < 1 or self.output_policy not in OUTPUT_LIMITS:
            raise ValueError(f"Invalid runtime: timeout {self.timeout}, threads {self.threads}, output policy {self.output_policy!r}")

    @property
    def contract(self) -> tuple:
        """What the agent-facing text depends on besides the budget (a client and its runner must agree on it)."""
        return (type(self).__qualname__, self.id, self.input_path, self.output_name, self.publication)

    @property
    def output_limit(self) -> int:
        from src.experiments.sandbox import OUTPUT_LIMITS

        return OUTPUT_LIMITS[self.output_policy]

    def input_lines(self, workspace: str = WORKSPACE) -> list[str]:
        """Part 1 "Input data": the run-time input sentences after the files mounted for reading."""
        return ["- At run time: the harness later runs solver.py in a separate attempt directory, which is its working directory, and solver.py must read its input from "
                f"{self.input_path} relative to that working directory. The {workspace} paths of this workspace do not exist at run time, so solver.py must not use them."]

    def output_section(self) -> str:
        """Part 1 "Output file"."""
        return ("## Output file\n"
                f"At run time solver.py must write its result to {self.output_name} in its working directory. The harness reads only that file: a missing or misplaced {self.output_name} is an execution failure. "
                "Do not execute the generated solver. The harness executes it later in its documented runtime directory.")

    def harness_rules(self) -> str:
        """The harness rules closing Part 2 "Rules" (also appended to the pasted generation prompt)."""
        from src.agent.prompts.builder import STANDALONE_RULE, budget_rule

        return STANDALONE_RULE + "\n" + budget_rule(self.timeout, self.threads) + ("\n" + self.publication if self.publication else "")


DEFAULT_RUNTIME = RuntimeSpec()
