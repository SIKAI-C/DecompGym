"""Review decomposition implementation through a fresh Codex session."""
from pathlib import Path

from src.agent.codex_client import CodexClient


def algorithm_review_prompt(suite: str) -> str:
    method = {"bd": "Benders decomposition", "cg": "column generation"}[suite]
    return (
        f"Inspect the supplied solver.py and determine whether it implements {method}. "
        "Check the master/subproblem split, dual or reduced-cost information, and iterative cut/column updates. "
        "A monolithic optimization model alone does not pass. "
        "Consider these possible implementation failures: decomposition functions or cut/column-generation code are defined but never called; the apparent decomposition loop does not use subproblem information to update the master; or the reported solution comes solely from a monolithic solve of the full original problem while the decomposition is unused or abandoned. "
        "Trace the entry point, relevant call paths, and the code that writes output.json. "
        "For conditional fallbacks, explain their triggering conditions and role from the source rather than assuming they ran. "
        "Distinguish these failures from genuine column generation followed by an integer solve of the restricted master over generated columns; that final integer solve is not by itself a reason to fail. "
        "Assess whether the requested decomposition substantively contributes to producing the solution, and cite concise code evidence in NOTES. "
        "Do not claim to have observed runtime behavior when reviewing source only. "
        "Do not execute or modify solver.py. Write alg_verifier.md with exactly a first line VERDICT: PASS or VERDICT: FAIL, followed by NOTES: and concise evidence. Feasibility is checked separately."
    )


class AlgorithmVerifier:
    @staticmethod
    def run_llm_verifier(attempt_dir: Path, client: CodexClient, suite: str) -> tuple[str | None, str]:
        prompt = algorithm_review_prompt(suite)
        target = attempt_dir / "alg_verifier.md"
        target.write_text("")
        result = client.edit_file(target, prompt, context=[attempt_dir / "solver.py"], log_dir=attempt_dir, step="llm_verifier")
        if not result.ok:
            return None, result.error or "Codex verifier failed"
        notes = target.read_text().strip()
        first = notes.splitlines()[0].strip().upper() if notes else ""
        return {"VERDICT: PASS": "pass", "VERDICT: FAIL": "fail"}.get(first), notes
