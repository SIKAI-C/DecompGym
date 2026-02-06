"""
Prompt builder for the agent. Responsible for building the prompt for the agent.
"""

from __future__ import annotations


class PromptBuilder:
    @staticmethod
    def build_reflection_prompt(base_prompt: str, gap_info: str | None = None) -> str:
        """
        Construct the follow-up prompt for aider after a failed attempt.
        The agent should read solver.py and exec.log (read-only), diagnose the failure,
        and write guidance for the next attempt into debug.md. Do not edit solver.py in this step.
        """
        gap_text = gap_info if gap_info is not None else "unknown"
        debug_context = [
            "You are making an attempt to solve the following problem.",
            base_prompt,
            "",
            "You have generated the solver.py file based on the problem description and the data.json file.",
            "And the exec.log file shows the output of the solver.py file.",
            "You also have alg_verifier.md with an LLM-based algorithm check; use it to improve the reflection.",
            f"The current objective gap to the optimal solution is: {gap_text}",
            "Your tasks:",
            "1) Read exec.log and solver.py to understand why the current attempt failed to execute or failed to find the optimal solution.\
                you need to reasoning the current solver.py and exec.log.",
            "2) Write a concise reflection to debug.md: what likely went wrong and what the next attempt should change",
            "3) Provide concrete tips for the following attempts, based on the reflection and write into the debug.md file.",
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
            "You have prior failed attempts with reflections (and you can read their solver.py, debug.md, summary.json):",
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