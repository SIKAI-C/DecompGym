"""Agent-facing text of the mounted-files protocol (prompt_protocol.context_delivery = "mounted-files-v1").

Without that setting (the paper and every older config) the context files are pasted into the prompt
(DockerCliClient.compose_prompt) and none of this module is used. Under CONTEXT_DELIVERY_MOUNTED:

- every context file is mounted read-only in the agent's Docker workspace (src.agent.mounted_context, layout below) and
  the prompt names each path, its size and its content instead of pasting it;
- the prompt is structured with the same headings, in the same order, for every family and every call
  (generation, regeneration, reflection):

    # Part 1: Workspace and files      (client side, workspace_part)
    ## File to modify | ## Input data | ## Output file | ## Files from previous attempts | ## Nothing else is available
    # Part 2: Task                     (runner side, task_part)
    ## Problem | ## Data | ## Output | ## Method requirements | ## Rules
    # Part 3: Reflection               (reflection calls only, reflection_part)

Part 2 is the family's rendered prompt.txt cut into contiguous pieces and regrouped under the headings; no task or
contract sentence is reworded or dropped (task_sections). The pieces: everything before the family's data marker
(DATA_START) is the Problem; from the marker up to "Task:" the Data; the "Task:" block the Output, except its two
leading rule sentences (TASK_RULES), which open the Rules; the method header and the suite paragraph of the contract
the Method requirements; the contract's common paragraphs (src.agent.prompts.method.COMMON_REQUIREMENTS) and the
harness rules (standalone solver, budget) the rest of the Rules. The pasted protocol's location note is replaced by
Part 1's Input data and Output file sections.

Layout (WORKSPACE = /work, the agent's working directory):
    /work/<target>                               the one writable file (solver.py or debug.md); the only file collected
    /work/instance/data.json                     the instance data
    /work/history/attempt_NN/<name>              files of earlier attempts (HISTORY_FILES)
"""
from __future__ import annotations

import re
from pathlib import Path

from src.agent.prompts.method import CONTRACTS, COMMON_REQUIREMENTS, METHOD_HEADER, method_requirements, require_contract
from src.agent.prompts.policy import edit_instruction
from src.utils.errors import ProtocolError


CONTEXT_DELIVERY_MOUNTED = "mounted-files-v1"
CONTEXT_DELIVERIES = (None, CONTEXT_DELIVERY_MOUNTED)
# The per-call workspace record of a mounted-files call (record_workspace_prompt `version`): its note opens Part 1.
MOUNTED_WORKSPACE_POLICY = "existing-target-cwd-mounted-files-v1"

WORKSPACE = "/work"
INSTANCE_DIR = WORKSPACE + "/instance"
HISTORY_DIR = WORKSPACE + "/history"
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
    """(path inside the agent container, description) of one context file of the paired runner."""
    path = Path(path)
    match = ATTEMPT_DIRECTORY.fullmatch(path.parent.name)
    if match and path.name in HISTORY_FILES:
        return f"{HISTORY_DIR}/{path.parent.name}/{path.name}", HISTORY_FILES[path.name].format(n=int(match[1]))
    if path.name == "data.json" and not match:
        return f"{INSTANCE_DIR}/data.json", INSTANCE_FILE
    raise ProtocolError(f"No mounted location for context file {path.parent.name}/{path.name}")


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


def task_part(prompt: str, *, suite: str, problem: str, contract: str, harness_rules: str, opening: str | None = None) -> str:
    """Part 2 of a mounted-files prompt: the prompt.txt pieces under the fixed headings, then the harness rules."""
    sections = task_sections(prompt, suite=suite, problem=problem, contract=contract)
    sections["Rules"].append(harness_rules)
    blocks = ["# Part 2: Task", *([opening] if opening else [])]
    blocks += [f"## {heading}\n" + "\n\n".join(sections[heading]) for heading in TASK_HEADINGS]
    return "\n\n".join(blocks)


def reflection_part(instructions: list[str], closing: str) -> str:
    """Part 3 of a mounted-files reflection: the reflection task lines of the pasted protocol, unchanged."""
    return "# Part 3: Reflection\n" + "\n".join(instructions) + "\n" + closing


def workspace_part(target: str, files: list, backend: str) -> tuple[str, str]:
    """(prompt to pass to record_workspace_prompt, its note): Part 1 for the writable /work/<target> and the mounted files.

    `files` are src.agent.mounted_context.MountedFile values (path, size, description), in context order. The
    returned prompt starts with the legacy "Edit only <target>." prefix, which record_workspace_prompt replaces by
    the note (the Part 1 heading, the File to modify heading and the paper's edit instruction)."""
    if Path(target).name != target or target not in {"solver.py", "debug.md"}:
        raise ValueError("A mounted-files call edits solver.py or debug.md")
    agy = backend == "agy-cli"
    writable = f"{WORKSPACE}/{target}"
    note = "# Part 1: Workspace and files\n\n## File to modify\n" + edit_instruction(target)
    prose = "Keep prose paragraphs on one physical line." if agy else "Keep prose paragraphs and list items on one physical line; rely on editor soft wrapping."
    modify = (f"Edit only {target}. The only file you may create or modify is {writable}; every other path in this workspace is read-only context. "
              f"Only {writable} is collected after this call: anything written elsewhere is ignored. {prose}")
    instance = [item for item in files if item.path.startswith(INSTANCE_DIR + "/")]
    history = [item for item in files if item.path.startswith(HISTORY_DIR + "/")]
    if len(instance) != 1 or instance[0].path != INSTANCE_DIR + "/data.json" or len(instance) + len(history) != len(files):
        raise ProtocolError("A mounted-files call needs exactly the instance data.json plus history files")
    data = instance[0]
    input_data = ("## Input data\n"
                  f"- For reading during this call: {data.path} ({data.size} bytes), {data.description}, mounted read-only.\n"
                  "- At run time: the harness later runs solver.py in a separate attempt directory, which is its working directory, and solver.py must read its input from ../../data.json relative to that working directory. "
                  f"The {WORKSPACE} paths of this workspace do not exist at run time, so solver.py must not use them.")
    output = ("## Output file\n"
              "At run time solver.py must write its result to output.json in its working directory. The harness reads only that file: a missing or misplaced output.json is an execution failure. "
              "Do not execute the generated solver. The harness executes it later in its documented runtime directory.")
    attempts: dict[str, list] = {}
    for item in history:
        attempts.setdefault(item.path.split("/")[3], []).append(item)
    review = target == "debug.md"
    if len(attempts) > 1 and review:
        raise ProtocolError("A reflection reviews exactly one attempt")
    lines = ["## Files from previous attempts"]
    if not attempts:
        lines.append("No files from previous attempts are provided.")
    for directory, items in attempts.items():
        number = int(ATTEMPT_DIRECTORY.fullmatch(directory)[1])
        lines.append(f"- Attempt {number}{' (the attempt under review)' if review else ''}, mounted read-only in {HISTORY_DIR}/{directory}/:")
        lines += [f"  - {item.path} ({item.size} bytes): {item.description}." for item in items]
    nothing = ("## Nothing else is available\n"
               f"Apart from the files listed above and {writable}, nothing else is available: do not inspect other directories, use external tools or services, or search for reference implementations."
               + (" Do not use memory, skills, subagents or prior conversations." if agy else ""))
    return "\n\n".join([modify, input_data, output, "\n".join(lines), nothing]), note
