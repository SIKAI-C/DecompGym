"""What an agent call reads: the ContextSpec of the paired runner (read-only artifacts, their roles and transport).

A ContextSpec has a versioned `id` and four independent parts:
- transport: PASTED (the paper: the files follow the prompt, DockerCliClient.compose_prompt) or MOUNTED (read-only
  files in the agent workspace, src.agent.mounted_context, recorded as prompt_protocol.context_delivery);
- representation of the instance data: FULL (data.json), PREVIEW (data.json, or above the threshold its structural
  preview instead; prompt_protocol.data_context = full-or-structural-preview-512KiB-v1) or BOTH (data.json and, above
  the threshold, also its preview). Any representation composes with either transport;
- history: the files of earlier attempts each call receives (HistorySets; PAPER_HISTORY is requirement 17's paper
  sets: the first attempt and every plain regeneration get data.json only, a reflection-policy regeneration also every
  earlier attempt's solver.py, debug.md and reflection_feedback.md, a reflection the reviewed attempt's solver.py,
  exec.log, alg_verifier.md and output.json);
- layout: the declared mount roles (src.agent.specs.MountLayout) of a mounted call.
The paired runner keeps the plain/reflection gate itself: a first attempt and every plain regeneration receive no file
of an earlier attempt, whatever the spec returns (PairedRunner.run_attempt).

The configs keep their keys and ids: context_spec(data_context, context_delivery) returns the spec of each reviewed
setting (LEGACY_CONTEXTS), whose ids are today's ids, and whose records (request fingerprints, case identity) are
exactly those of the earlier fixed implementation. Any other spec is constructed explicitly; the runner then records
its id (src.experiments.paired_runner.PairedRunner.spec_record). No config selects one.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.agent.prompts.mounted import CONTEXT_DELIVERIES, CONTEXT_DELIVERY_MOUNTED
from src.agent.specs import MOUNT_LAYOUT, PREVIEW_LAYOUT, PREVIEW_ROLE, SOURCES_LAYOUT, MountLayout
from src.experiments.data_context import DATA_CONTEXT_BOTH, DATA_CONTEXT_POLICIES, DATA_CONTEXT_PREVIEW, data_context_files

PASTED, MOUNTED = "pasted", "mounted"
FULL, PREVIEW, BOTH = "full", "preview", "both"
REPRESENTATIONS = {FULL: None, PREVIEW: DATA_CONTEXT_PREVIEW, BOTH: DATA_CONTEXT_BOTH}


def _is_pattern(name: str) -> bool:
    return any(character in name for character in "*?[")


@dataclass(frozen=True)
class HistorySets:
    """The history files of each call kind as attempt-relative names or glob patterns (for example cpp/*.cpp), in this
    order. A missing named file is skipped by the clients; a pattern contributes its existing regular files, sorted."""
    id: str
    regeneration: tuple[str, ...]  # of every earlier attempt, for a reflection-policy regeneration only
    reflection: tuple[str, ...]  # of the attempt under review

    @staticmethod
    def _expand(directory: Path, name: str) -> list[Path]:
        return sorted(path for path in directory.glob(name) if path.is_file() and not path.is_symlink()) if _is_pattern(name) else [directory / name]

    def previous_attempts(self, attempts_root: Path, current_idx: int) -> list[Path]:
        files = []
        for idx in range(1, current_idx):
            directory = attempts_root / f"attempt_{idx:02d}"
            files.extend(path.resolve() for name in self.regeneration for path in self._expand(directory, name) if path.exists())
        return files

    def reviewed(self, attempt_dir: Path) -> list[Path]:
        return [path for name in self.reflection for path in self._expand(attempt_dir, name)]


PAPER_HISTORY = HistorySets("paper-history-sets-v1", ("solver.py", "debug.md", "reflection_feedback.md"), ("solver.py", "exec.log", "alg_verifier.md", "output.json"))
# The paper sets plus a multi-file delivery's C++ sources (no config uses them).
SOURCES_HISTORY = HistorySets("paper-history-sets-with-sources-v1", ("solver.py", "cpp/*.cpp", "cpp/*.hpp", "debug.md", "reflection_feedback.md"),
                              ("solver.py", "cpp/*.cpp", "cpp/*.hpp", "exec.log", "alg_verifier.md", "output.json"))


@dataclass(frozen=True)
class ContextSpec:
    id: str
    transport: str = PASTED
    representation: str = FULL
    history: HistorySets = PAPER_HISTORY
    layout: MountLayout = MOUNT_LAYOUT

    def __post_init__(self):
        if self.transport not in (PASTED, MOUNTED) or self.representation not in REPRESENTATIONS:
            raise ValueError(f"Unknown context transport or representation: {self.transport!r}, {self.representation!r}")
        if self.mounted and self.representation != FULL and PREVIEW_ROLE not in self.layout.roles:
            raise ValueError("A mounted preview needs a layout that declares the data_preview role")

    @property
    def mounted(self) -> bool:
        return self.transport == MOUNTED

    @property
    def data_context(self) -> str | None:
        """The data-context id (the recorded prompt_protocol.data_context; None: the full data.json)."""
        return REPRESENTATIONS[self.representation]

    @property
    def context_delivery(self) -> str | None:
        """The context-delivery id a client of this spec carries (prompt_protocol.context_delivery; None: pasted)."""
        return CONTEXT_DELIVERY_MOUNTED if self.mounted else None

    def instance_files(self, instance: Path) -> tuple[list[Path], dict | None]:
        """The instance-data entries of a generation or reflection context and their evidence record."""
        return data_context_files(instance, self.data_context)

    def generation(self, instance: Path, attempts_root: Path, idx: int, mode: str) -> tuple[list[Path], dict | None]:
        """A generation's context: the instance data, plus (reflection policy, idx > 1) every earlier attempt's history
        files. The plain policy never receives history: that is the meaning of plain regeneration."""
        files, record = self.instance_files(instance)
        if mode == "reflection" and idx > 1:
            files += self.history.previous_attempts(attempts_root, idx)
        return files, record

    def reflection(self, instance: Path, attempt_dir: Path) -> tuple[list[Path], dict | None]:
        files, record = self.instance_files(instance)
        return files + self.history.reviewed(attempt_dir), record


# The reviewed settings: (prompt_protocol.data_context, prompt_protocol.context_delivery) -> spec.
PAPER_CONTEXT = ContextSpec("pasted-full-data-v1")
PREVIEW_CONTEXT = ContextSpec(DATA_CONTEXT_PREVIEW, representation=PREVIEW)
MOUNTED_CONTEXT = ContextSpec(CONTEXT_DELIVERY_MOUNTED, transport=MOUNTED)
LEGACY_CONTEXTS = {(None, None): PAPER_CONTEXT, (DATA_CONTEXT_PREVIEW, None): PREVIEW_CONTEXT, (None, CONTEXT_DELIVERY_MOUNTED): MOUNTED_CONTEXT}
# Expressible, not used by any config: the mounted full data.json plus its structural preview (large-scale option).
MOUNTED_WITH_PREVIEW = ContextSpec(f"{CONTEXT_DELIVERY_MOUNTED}+{DATA_CONTEXT_BOTH}", transport=MOUNTED, representation=BOTH, layout=PREVIEW_LAYOUT)
# Expressible, not used by any config: mounted context whose history includes a multi-file delivery's C++ sources.
MOUNTED_WITH_SOURCES = ContextSpec(f"{CONTEXT_DELIVERY_MOUNTED}+{SOURCES_HISTORY.id}", transport=MOUNTED, history=SOURCES_HISTORY, layout=SOURCES_LAYOUT)


def context_spec(data_context: str | None = None, context_delivery: str | None = None) -> ContextSpec:
    """The ContextSpec of a reviewed (data_context, context_delivery) setting; ValueError for any other pair."""
    if data_context not in DATA_CONTEXT_POLICIES:
        raise ValueError(f"Unknown data-context policy: {data_context}")
    if context_delivery not in CONTEXT_DELIVERIES:
        raise ValueError(f"Unknown context delivery: {context_delivery}")
    if (data_context, context_delivery) not in LEGACY_CONTEXTS:
        raise ValueError("The mounted-files protocol delivers the complete data.json; it has no data-context preview")
    return LEGACY_CONTEXTS[(data_context, context_delivery)]
