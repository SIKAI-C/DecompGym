"""Generation and reflection prompts built around each instance's problem prompt."""

from __future__ import annotations

from src.agent.prompts.policy import OBJECTIVE_RETRY


LOCATION_NOTE = (
    "The data below are provided as read-only context for this editing call.\n"
    "The generated solver.py will be executed later in a separate runtime directory, where it must read ../../data.json and write output.json in that runtime directory."
)


class PromptBuilder:
    """Build prompts for agent generation and reflection workflows."""
    @staticmethod
    def build_generation_prompt(problem_prompt: str, *, solver_timeout: int, solver_threads: int) -> str:
        """The instance prompt plus the runtime location and the solver budget."""
        prompt = PromptBuilder.ensure_location_prompt(problem_prompt)
        prompt += "\n\nImplement a standalone solver.py using gurobipy, NumPy, and the Python standard library. Write output.json using the documented contract. Do not import repository implementations or load reference solutions."
        prompt += f"\nThe harness limits the entire solver.py execution to {solver_timeout} seconds and numerical solvers to {solver_threads} thread(s). Design your algorithm and termination handling within this budget."
        return prompt

    @staticmethod
    def build_reflection_prompt(base_prompt: str, *, objective_quality_failed: bool = False) -> str:
        """
        Construct the follow-up prompt after a failed attempt.
        The agent should read solver.py and exec.log (read-only), diagnose the failure,
        and write guidance for the next attempt into debug.md. Do not edit solver.py in this step.
        """
        if type(objective_quality_failed) is not bool:
            raise TypeError("Objective feedback accepts only a boolean, never a numeric gap")
        debug_context = [
            "You are making an attempt to solve the following problem.",
            base_prompt,
            "",
            "You have generated the solver.py file based on the problem description and the data.json file.",
            "And the exec.log file shows the output of the solver.py file.",
            "You also have alg_verifier.md with an LLM-based algorithm check; use it to improve the reflection.",
            *([OBJECTIVE_RETRY] if objective_quality_failed else []),
            "Your tasks:",
            "1) Read solver.py, exec.log, and the verifier feedback to diagnose execution, feasibility, algorithm, or objective-quality failures.",
            "2) Write a concise reflection to debug.md: what likely went wrong and what the next attempt should change",
            "3) List the concrete changes the next attempt should make in debug.md.",
        ]
        return "\n".join(debug_context) + "\nKeep the reflection concise and actionable."

    @staticmethod
    def ensure_location_prompt(base_prompt: str) -> str:
        """
        Ensure the prompt includes the location note for data/prompt/output files.
        """
        if LOCATION_NOTE in base_prompt:
            return base_prompt
        return base_prompt.rstrip() + "\n\n" + LOCATION_NOTE
