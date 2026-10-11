"""
Prompt builder for the agent. Responsible for building the prompt for the agent.
"""

from __future__ import annotations

from src.agent.prompt_policy import OBJECTIVE_RETRY


class PromptBuilder:
    """Build prompts for agent generation and reflection workflows."""
    @staticmethod
    def build_reflection_prompt(base_prompt: str, *, objective_quality_failed: bool = False) -> str:
        """
        Construct the follow-up prompt for Codex after a failed attempt.
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
        return "\n".join(debug_context)

    @staticmethod
    def build_prev_debug_prompt(base_prompt: str, prior_notes: str) -> str:
        """
        Prompt used before generating a new solver attempt, surfacing lessons from previous failures.
        """
        context = [
            base_prompt,
            "",
            "You have prior failed attempts with reflections (and you can read their solver.py and debug.md):",
            prior_notes or "(no prior reflections)",
            "",
            "Incorporate the lessons above to avoid repeating mistakes. Use the prior files as read-only context.",
        ]
        return "\n".join(context)

    @staticmethod
    def ensure_location_prompt(base_prompt: str) -> str:
        """
        Ensure the prompt includes the location note for data/prompt/output files.
        """
        location_note = (
            "IMPORTANT: data.json and prompt.txt are located two levels up (../../). "
            "Read data from ../../data.json and ../../prompt.txt, and write output.json in the current folder. "
            "Do NOT read from ../data.json or attempts/data.json.\n\n"
        )
        if location_note in base_prompt:
            return base_prompt
        return base_prompt.rstrip() + "\n\n" + location_note
