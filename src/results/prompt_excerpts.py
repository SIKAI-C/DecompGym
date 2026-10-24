"""The paper's verbatim prompt examples, exported from the saved prompts of one recorded pair.

The example is Claude Sonnet 5 on csp_medium_seed000 in the Claude extension campaign. Long context blocks are
replaced by `<<elided: N lines of ...>>`, the status requirement is wrapped in `@@...@@` (printed in red by the paper),
and non-ASCII dashes become `--` for the LaTeX listings.
"""
from __future__ import annotations

import json
from pathlib import Path
import re

from src.agent.prompts.method import METHOD_HEADER, METHOD_REQUIREMENTS
from src.agent.prompts.policy import OBJECTIVE_RETRY


EXAMPLE = {"run": "fixed81_strict_claude_extension_20261014_r1", "model": "claude-sonnet-5", "suite": "cg", "problem": "csp", "instance": "csp_medium_seed000"}
STATUS_SENTENCE = "Use status feasible when returning an incumbent without a global optimality proof; optimality of an integer restricted master alone does not prove global optimality."
BLOCK = re.compile(r"--- BEGIN CONTEXT: (?P<label>[^\n]+) ---\n(?P<body>.*?)\n--- END CONTEXT: (?P=label) ---", re.S)
SHOWN_ITEM_TYPES = 2


def elided(match):
    # Line counts follow `wc -l`, so a file without a final newline counts one line fewer.
    label, body = match["label"], match["body"]
    return f"--- BEGIN CONTEXT: {label} ---\n<<elided: {body.count(chr(10))} lines of {label}>>\n--- END CONTEXT: {label} ---"


def blocks(text):
    return "\n\n".join(elided(m) for m in BLOCK.finditer(text))


def highlight(text):
    assert text.count(STATUS_SENTENCE) == 1
    return text.replace(STATUS_SENTENCE, "@@" + STATUS_SENTENCE + "@@")


def ascii_dashes(text):
    return text.replace("—", "--")


def short_data(match):
    """Show the first item types of the cutting-stock data and elide the rest."""
    data = json.loads(match["body"])
    items = data["item_types"]
    shown = json.dumps(data | {"item_types": items[:SHOWN_ITEM_TYPES]}, indent=2).splitlines()
    closing = shown.index("  ]")
    shown.insert(closing, f"    <<elided: {len(items) - SHOWN_ITEM_TYPES} more item types>>")
    return f"--- BEGIN CONTEXT: {match['label']} ---\n" + "\n".join(shown) + f"\n--- END CONTEXT: {match['label']} ---"


def excerpts(experiments: Path) -> dict[str, str]:
    case = experiments / EXAMPLE["run"] / "runs" / EXAMPLE["model"]
    plain = case / "plain" / EXAMPLE["suite"] / EXAMPLE["problem"] / EXAMPLE["instance"] / "attempts"
    reflection = case / "reflection" / EXAMPLE["suite"] / EXAMPLE["problem"] / EXAMPLE["instance"] / "attempts"
    generation = (plain / "attempt_01/agent_prompt.md").read_text()
    judge = (plain / "attempt_01/llm_verifier_prompt.md").read_text()
    diagnosis = (reflection / "attempt_01/reflection_prompt.md").read_text()
    revision = (reflection / "attempt_02/agent_prompt.md").read_text()
    tasks = diagnosis[diagnosis.index("You have generated the solver.py file"):diagnosis.index("Keep the reflection concise and actionable.") + len("Keep the reflection concise and actionable.")]
    texts = {
        "generation_csp.txt": BLOCK.sub(short_data, highlight(generation)),
        "contract_benders.txt": METHOD_HEADER + "\n\n" + METHOD_REQUIREMENTS["bd"],
        "judge.txt": BLOCK.sub(elided, highlight(judge)),
        "judge_rationale_sonnet5.txt": (plain / "attempt_01/alg_verifier.md").read_text(),
        "quality_feedback.txt": OBJECTIVE_RETRY,
        "reflection_diagnosis.txt": "<<the workspace rules of Prompt 1 with debug.md as the file to edit, the line \"You are making an attempt to solve the following problem.\", and the text of Prompt 1 from its first task line through its runtime rules>>\n\n" + tasks + "\n\n" + blocks(diagnosis),
        "reflection_advice_sonnet5.txt": ascii_dashes((reflection / "attempt_01/debug.md").read_text()),
        "revision_generation.txt": "<<the generation instructions of Prompt 1, unchanged, followed by these context blocks>>\n\n" + blocks(revision),
    }
    return {name: text.rstrip("\n") + "\n" for name, text in texts.items()}


def write(experiments: Path, out: Path):
    directory = out / "prompts"
    directory.mkdir(parents=True, exist_ok=True)
    for name, text in excerpts(experiments).items():
        (directory / name).write_text(text)
    return sorted(p.name for p in directory.iterdir())
