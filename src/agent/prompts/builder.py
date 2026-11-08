"""Generation and reflection prompts built around each instance's problem prompt."""

from __future__ import annotations

from src.agent.prompts.policy import OBJECTIVE_RETRY


LOCATION_NOTE = (
    "The data below are provided as read-only context for this editing call.\n"
    "The generated solver.py will be executed later in a separate runtime directory, where it must read ../../data.json and write output.json in that runtime directory."
)
STANDALONE_RULE = "Implement a standalone solver.py using gurobipy, NumPy, and the Python standard library. Write output.json using the documented contract. Do not import repository implementations or load reference solutions."
REFLECTION_OPENING = "You are making an attempt to solve the following problem."
REFLECTION_CLOSING = "Keep the reflection concise and actionable."


def memory_budget(memory_mib: int) -> str:
    return f"{memory_mib // 1024} GiB" if memory_mib % 1024 == 0 else f"{memory_mib} MiB"


def budget_rule(solver_timeout: int, solver_threads: int, memory_mib: int | None = None) -> str:
    """The solver budget sentence; with a memory budget (only the lanes variant's RuntimeSpec has one) one more clause."""
    memory = "" if memory_mib is None else f" The solver.py process and its child processes share a resident-memory budget of {memory_budget(memory_mib)}; a run that exceeds it is stopped."
    return f"The harness limits the entire solver.py execution to {solver_timeout} seconds and numerical solvers to {solver_threads} thread(s).{memory} Design your algorithm and termination handling within this budget."


def reflection_instructions(objective_quality_failed: bool) -> list[str]:
    """The reflection task's lines (the same in the pasted and the mounted-files protocol)."""
    if type(objective_quality_failed) is not bool:
        raise TypeError("Objective feedback accepts only a boolean, never a numeric gap")
    return [
        "You have generated the solver.py file based on the problem description and the data.json file.",
        "And the exec.log file shows the output of the solver.py file.",
        "You also have alg_verifier.md with an LLM-based algorithm check; use it to improve the reflection.",
        *([OBJECTIVE_RETRY] if objective_quality_failed else []),
        "Your tasks:",
        "1) Read solver.py, exec.log, and the verifier feedback to diagnose execution, feasibility, algorithm, or objective-quality failures.",
        "2) Write a concise reflection to debug.md: what likely went wrong and what the next attempt should change",
        "3) List the concrete changes the next attempt should make in debug.md.",
    ]


class PromptBuilder:
    """Build prompts for agent generation and reflection workflows."""
    @staticmethod
    def build_generation_prompt(problem_prompt: str, *, solver_timeout: int, solver_threads: int) -> str:
        """The instance prompt plus the runtime location and the solver budget."""
        prompt = PromptBuilder.ensure_location_prompt(problem_prompt)
        prompt += "\n\n" + STANDALONE_RULE
        prompt += "\n" + budget_rule(solver_timeout, solver_threads)
        return prompt

    @staticmethod
    def build_reflection_prompt(base_prompt: str, *, objective_quality_failed: bool = False) -> str:
        """
        Construct the follow-up prompt after a failed attempt.
        The agent should read solver.py and exec.log (read-only), diagnose the failure,
        and write guidance for the next attempt into debug.md. Do not edit solver.py in this step.
        """
        instructions = reflection_instructions(objective_quality_failed)
        return "\n".join([REFLECTION_OPENING, base_prompt, "", *instructions]) + "\n" + REFLECTION_CLOSING

    @staticmethod
    def build_structured_generation_prompt(problem_prompt: str, *, suite: str, problem: str, contract: str, solver_timeout: int, solver_threads: int) -> str:
        """Part 2 (the task) of a mounted-files generation prompt from a stored prompt.txt; the client adds Part 1 (src.agent.prompts.mounted)."""
        from src.agent.prompts.mounted import task_sections
        from src.agent.specs import RuntimeSpec

        return PromptBuilder.structured_generation(task_sections(problem_prompt, suite=suite, problem=problem, contract=contract), runtime=RuntimeSpec(solver_timeout, solver_threads))

    @staticmethod
    def build_structured_reflection_prompt(problem_prompt: str, *, suite: str, problem: str, contract: str, solver_timeout: int, solver_threads: int,
                                           objective_quality_failed: bool = False) -> str:
        """Part 2 (the task, opened as in the pasted reflection) and Part 3 (the reflection task) of a mounted-files reflection."""
        from src.agent.prompts.mounted import task_sections
        from src.agent.specs import RuntimeSpec

        return PromptBuilder.structured_reflection(task_sections(problem_prompt, suite=suite, problem=problem, contract=contract), runtime=RuntimeSpec(solver_timeout, solver_threads),
                                                   objective_quality_failed=objective_quality_failed)

    @staticmethod
    def structured_generation(sections: dict[str, list[str]], *, runtime) -> str:
        """Part 2 of a mounted-files generation from explicit sections (any adapter) and the RuntimeSpec's harness rules."""
        from src.agent.prompts.mounted import render_task_part

        return render_task_part(sections, harness_rules=runtime.harness_rules())

    @staticmethod
    def structured_reflection(sections: dict[str, list[str]], *, runtime, objective_quality_failed: bool = False) -> str:
        """Parts 2 and 3 of a mounted-files reflection from explicit sections."""
        from src.agent.prompts.mounted import reflection_part, render_task_part

        instructions = reflection_instructions(objective_quality_failed)
        return render_task_part(sections, harness_rules=runtime.harness_rules(), opening=REFLECTION_OPENING) + "\n\n" + reflection_part(instructions, REFLECTION_CLOSING)

    @staticmethod
    def ensure_location_prompt(base_prompt: str) -> str:
        """
        Ensure the prompt includes the location note for data/prompt/output files.
        """
        if LOCATION_NOTE in base_prompt:
            return base_prompt
        return base_prompt.rstrip() + "\n\n" + LOCATION_NOTE
