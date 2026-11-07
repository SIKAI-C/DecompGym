"""Agent-facing text of the mounted-files protocol (prompt_protocol.context_delivery = "mounted-files-v1").

Without that setting (the paper and every older config) the context files are pasted into the prompt
(DockerCliClient.compose_prompt) and none of this module is used. Under CONTEXT_DELIVERY_MOUNTED:

- every context file is mounted read-only in the agent's Docker workspace (src.agent.mounted_context, layout below) and
  the prompt names each path, its size and its content instead of pasting it;
- the prompt is structured with the same headings, in the same order, for every family and every call
  (generation, regeneration, reflection):

    # Part 1: Workspace and files      (client side, workspace_part)
    ## File to modify | ## Input data | ## Output file | ## Files from previous attempts | ## No other task files
    # Part 2: Task                     (runner side, render_task_part)
    ## Problem | ## Data | ## Output | ## Method requirements | ## Rules
    # Part 3: Reflection               (reflection calls only, reflection_part)

Part 1 is rendered from three independent specs (src.agent.specs): the MountLayout (which files, which roles), the
DeliverySpec (the writable and collected target: "File to modify", "No other task files") and the RuntimeSpec
(the run-time input and output paths: "Input data", "Output file"). The defaults reproduce the earlier fixed text.

Part 2 renders explicit sections (TASK_HEADINGS -> list of text blocks) and the RuntimeSpec's harness rules. A section
adapter supplies them: PROSE_SECTIONS, the small-scale adapter, cuts the family's rendered prompt.txt into contiguous
pieces and regroups them under the headings; no task or contract sentence is reworded or dropped (task_sections). The
pieces: everything before the family's data marker (DATA_START) is the Problem; from the marker up to "Task:" the Data;
the "Task:" block the Output, except its two leading rule sentences (TASK_RULES), which open the Rules; the method header
and the suite paragraph of the contract the Method requirements; the contract's common paragraphs
(src.agent.prompts.method.COMMON_REQUIREMENTS) and the harness rules (standalone solver, budget) the rest of the Rules.
The pasted protocol's location note is replaced by Part 1's Input data and Output file sections. A family track may
instead supply its sections directly (DirectSections), checked against the same headings and the campaign's contract.

Layout (WORKSPACE = /work, the agent's working directory; src.agent.specs.MOUNT_LAYOUT):
    /work/<target>                               the one writable file (solver.py or debug.md); the only file collected
    /work/instance/data.json                     the instance data
    /work/history/attempt_NN/<name>              files of earlier attempts (HISTORY_FILES)
"""
from __future__ import annotations

import json
from pathlib import Path

from src.agent.prompts.method import CONTRACTS, COMMON_REQUIREMENTS, METHOD_HEADER, method_requirements, require_contract
from src.agent.prompts.policy import edit_instruction
# The layout constants live with the specs (src.agent.specs); they are re-exported here for the earlier importers.
from src.agent.specs import (ATTEMPT_DIRECTORY, CONTEXT_DIRECTORIES, DEFAULT_RUNTIME, HISTORY_DIR, HISTORY_FILES, HISTORY_SECTION, INPUT_SECTION, INSTANCE_DIR,  # noqa: F401
                             INSTANCE_FILE, MOUNT_LAYOUT, SINGLE_FILE_DELIVERY, WORKSPACE)
from src.utils.errors import ProtocolError


CONTEXT_DELIVERY_MOUNTED = "mounted-files-v1"
CONTEXT_DELIVERIES = (None, CONTEXT_DELIVERY_MOUNTED)
# The per-call workspace record of a mounted-files call (record_workspace_prompt `version`): its note opens Part 1.
MOUNTED_WORKSPACE_POLICY = "existing-target-cwd-mounted-files-v1"
# The same record for a mounted call that may also write C++ sources (multi-file delivery; no config uses it).
MOUNTED_CPP_WORKSPACE_POLICY = "existing-target-cwd-mounted-files-cpp-v1"

# Part 2 pieces. Every rendered prompt.txt has this Task block opening and ends with the contract text.
TASK_LABEL = "Task:"
TASK_RULES = "Assume gurobipy and NumPy are available; the Python standard library is also allowed. Do not depend on any files other than data.json."
TASK_LEAD = "Write a Python program solver.py that:"
# The line at which each family's Data section starts (its first line about this instance's data or data.json).
DATA_START = {
    "ufl": "- There are ",
    "km": "The data for this instance lives in data.json with fields:",
    "fctp": "The data for this instance is in data.json with fields:",
    "scfl": "This instance has ",
    "lbbd_ps": "This instance has ",
    "tsso": "The JSON file data.json contains:",
    "lrp": "The instance is stored in data.json with fields:",
    "lbbd": "This instance has ",
    "bp": "The input file data.json provides:",
    "csp": "Input file data.json contains:",
    "gap": "Input data.json contains:",
    "bppc": "This instance has ",
    "clrp": "This instance has ",
    "cvrp": "This instance has ",
    "vrptw": "This instance has ",
    "pdptw": "This instance has ",
    "rcsp": "This instance has ",
    "shift": "Input data.json contains:",
    "gcol": "This instance has ",
    "nrp": "This instance has ",
}
TASK_HEADINGS = ("Problem", "Data", "Output", "Method requirements", "Rules")


def mount_location(path: Path) -> tuple[str, str]:
    """(path inside the agent container, description) of one context file of the paired runner (MOUNT_LAYOUT)."""
    location, description, _ = MOUNT_LAYOUT.locate(path)
    return location, description


def task_sections(prompt: str, *, suite: str, problem: str, contract: str) -> dict[str, list[str]]:
    """The rendered prompt.txt cut into the Part 2 sections (each a list of verbatim text blocks, in order)."""
    contract = require_contract(contract)
    method = method_requirements(suite, contract)
    if prompt.count(method) != 1 or not prompt.rstrip("\n").endswith(method):
        raise ProtocolError(f"{suite}/{problem}: prompt.txt does not end with the {contract} method contract")
    suite_text = CONTRACTS[contract][suite]
    if method != "\n\n".join((METHOD_HEADER, suite_text, *COMMON_REQUIREMENTS)):
        raise ProtocolError("Unexpected method contract layout")
    head = prompt[:prompt.index(method)]
    opening = "\n" + TASK_LABEL + "\n" + TASK_RULES + " " + TASK_LEAD + "\n"
    if head.count(opening) != 1:
        raise ProtocolError(f"{suite}/{problem}: prompt.txt has no single Task block opening")
    description, output = head.split(opening)
    if problem not in DATA_START:
        raise ProtocolError(f"No Part 2 data marker for family {problem!r}")
    marker = "\n" + DATA_START[problem]
    if marker not in description:
        raise ProtocolError(f"{suite}/{problem}: data marker {DATA_START[problem]!r} not found")
    cut = description.index(marker) + 1
    sections = {"Problem": [description[:cut].strip("\n")], "Data": [description[cut:].strip("\n")],
                "Output": [TASK_LABEL + "\n" + TASK_LEAD + "\n" + output.strip("\n")],
                "Method requirements": [METHOD_HEADER + "\n\n" + suite_text], "Rules": [TASK_RULES, *COMMON_REQUIREMENTS]}
    if not all(block.strip() for blocks in sections.values() for block in blocks):
        raise ProtocolError(f"{suite}/{problem}: an empty Part 2 section")
    return sections


def check_sections(sections, *, suite: str, problem: str, contract: str) -> dict[str, list[str]]:
    """Explicit Part 2 sections (from any adapter): exactly TASK_HEADINGS, each a non-empty list of non-empty text blocks;
    the Method requirements open with the campaign contract's header and suite paragraph and the Rules hold its common
    paragraphs, so that author, reflection and judge read the same contract."""
    contract = require_contract(contract)
    if not isinstance(sections, dict) or set(sections) != set(TASK_HEADINGS):
        raise ProtocolError(f"{suite}/{problem}: Part 2 needs exactly the sections {TASK_HEADINGS}")
    if not all(isinstance(blocks, (list, tuple)) and blocks and all(isinstance(block, str) and block.strip() for block in blocks) for blocks in sections.values()):
        raise ProtocolError(f"{suite}/{problem}: every Part 2 section is a non-empty list of non-empty text blocks")
    checked = {heading: list(sections[heading]) for heading in TASK_HEADINGS}
    if checked["Method requirements"][0] != METHOD_HEADER + "\n\n" + CONTRACTS[contract][suite] or not all(paragraph in checked["Rules"] for paragraph in COMMON_REQUIREMENTS):
        raise ProtocolError(f"{suite}/{problem}: Part 2 does not carry the {contract} method contract")
    return checked


def render_task_part(sections: dict[str, list[str]], *, harness_rules: str, opening: str | None = None) -> str:
    """Part 2 of a mounted-files prompt: the sections under the fixed headings, the harness rules closing the Rules."""
    sections = {heading: list(sections[heading]) for heading in TASK_HEADINGS}
    sections["Rules"].append(harness_rules)
    blocks = ["# Part 2: Task", *([opening] if opening else [])]
    blocks += [f"## {heading}\n" + "\n\n".join(sections[heading]) for heading in TASK_HEADINGS]
    return "\n\n".join(blocks)


def task_part(prompt: str, *, suite: str, problem: str, contract: str, harness_rules: str, opening: str | None = None) -> str:
    """Part 2 of a mounted-files prompt from a stored prompt.txt (the prose adapter)."""
    return render_task_part(task_sections(prompt, suite=suite, problem=problem, contract=contract), harness_rules=harness_rules, opening=opening)


class ProseSections:
    """The small-scale section adapter: the Part 2 sections cut from the case's stored prompt.txt (task_sections)."""
    id = "prompt-txt-prose-sections-v1"
    inputs = ("prompt.txt",)  # case files the adapter reads (frozen in the case identity next to data.json)

    def __call__(self, instance: Path, *, suite: str, problem: str, contract: str) -> dict[str, list[str]]:
        return task_sections((Path(instance) / "prompt.txt").read_text(), suite=suite, problem=problem, contract=contract)


PROSE_SECTIONS = ProseSections()


class DirectSections:
    """A section adapter that builds the sections from the case's data.json (`render(data, suite=, problem=, contract=)`),
    for a family track whose task text is written as sections (for example a large-scale b_problem_xl); checked by
    check_sections."""

    inputs = ()

    def __init__(self, id: str, render):
        self.id, self.render = id, render

    def __call__(self, instance: Path, *, suite: str, problem: str, contract: str) -> dict[str, list[str]]:
        data = json.loads((Path(instance) / "data.json").read_text())
        return check_sections(self.render(data, suite=suite, problem=problem, contract=contract), suite=suite, problem=problem, contract=contract)


def reflection_part(instructions: list[str], closing: str) -> str:
    """Part 3 of a mounted-files reflection: the reflection task lines of the pasted protocol, unchanged."""
    return "# Part 3: Reflection\n" + "\n".join(instructions) + "\n" + closing


def workspace_note(target: str, instruction: str | None = None) -> str:
    """The note that opens Part 1 of a mounted-files call (record_workspace_prompt `note`): the Part 1 heading, the
    File to modify heading and the edit instruction (the paper's, unless the delivery adds to it)."""
    return "# Part 1: Workspace and files\n\n## File to modify\n" + (edit_instruction(target) if instruction is None else instruction)


def workspace_part(target: str, files: list, backend: str, *, delivery=SINGLE_FILE_DELIVERY, runtime=DEFAULT_RUNTIME, layout=MOUNT_LAYOUT) -> tuple[str, str]:
    """(prompt to pass to record_workspace_prompt, its note): Part 1 for the writable target(s) and the mounted files.

    `files` are src.agent.mounted_context.MountedFile values (path, size, description), in context order, placed by
    `layout`; `delivery` (src.agent.specs.DeliverySpec) states what may be written and is collected, `runtime`
    (src.agent.specs.RuntimeSpec) the run-time input and output contract. The returned prompt starts with the legacy
    "Edit only <target>." prefix, which record_workspace_prompt replaces by the note (the Part 1 heading, the File to
    modify heading and the edit instruction)."""
    if Path(target).name != target or target not in {"solver.py", "debug.md"}:
        raise ValueError("A mounted-files call edits solver.py or debug.md")
    agy = backend == "agy-cli"
    note = workspace_note(target, delivery.instruction(target))
    prose = "Keep prose paragraphs on one physical line." if agy else "Keep prose paragraphs and list items on one physical line; rely on editor soft wrapping."
    modify = f"Edit only {target}. {delivery.modify_text(target, WORKSPACE)} {prose}"
    instance = [item for item in files if layout.section(item.path) == INPUT_SECTION]
    history = [item for item in files if layout.section(item.path) == HISTORY_SECTION]
    if not instance or len(instance) + len(history) != len(files):
        raise ProtocolError("A mounted-files call needs exactly the instance data.json plus history files")
    input_data = "## Input data\n" + "\n".join([f"- For reading during this call: {item.path} ({item.size} bytes), {item.description}, mounted read-only." for item in instance]
                                                + runtime.input_lines(WORKSPACE))
    output = runtime.output_section()
    attempts: dict[tuple[str, str], list] = {}
    for item in history:
        directory = layout.directory(item.path)
        attempts.setdefault((directory, item.path[len(directory) + 1:].split("/")[0]), []).append(item)
    review = target == "debug.md"
    if len({directory for _, directory in attempts}) > 1 and review:  # one attempt, possibly across several mount directories
        raise ProtocolError("A reflection reviews exactly one attempt")
    lines = ["## Files from previous attempts"]
    if not attempts:
        lines.append("No files from previous attempts are provided.")
    for (root, directory), items in attempts.items():
        match = ATTEMPT_DIRECTORY.fullmatch(directory)
        if match is None:
            raise ProtocolError(f"History file outside an attempt directory: {items[0].path}")
        lines.append(f"- Attempt {int(match[1])}{' (the attempt under review)' if review else ''}, mounted read-only in {root}/{directory}/:")
        lines += [f"  - {item.path} ({item.size} bytes): {item.description}." for item in items]
    nothing = ("## No other task files\n"
               f"No other task files are provided: apart from the files listed above and {delivery.writable_text(target, WORKSPACE)}, the workspace contains no data, references or results of this or any other instance; do not inspect other directories, use external tools or services, or search for reference implementations."
               + (" Do not use memory, skills, subagents or prior conversations." if agy else ""))
    return "\n\n".join([modify, input_data, output, "\n".join(lines), nothing]), note
