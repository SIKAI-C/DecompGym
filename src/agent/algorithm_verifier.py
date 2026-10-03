"""Review decomposition implementation through a fresh Codex session."""
from pathlib import Path

from src.agent.codex_client import CodexClient


class AlgorithmVerifier:
    @staticmethod
    def run_llm_verifier(attempt_dir: Path, client: CodexClient, suite: str) -> tuple[str | None, str]:
        method = "Benders decomposition" if suite == "bd" else "column generation"
        prompt = f"Inspect the supplied solver.py and determine whether it implements {method}. Check the master/subproblem split, dual or reduced-cost information, and iterative cut/column updates. A monolithic optimization model alone does not pass. Do not execute or modify solver.py. Write alg_verifier.md with exactly a first line VERDICT: PASS or VERDICT: FAIL, followed by NOTES: and concise evidence. Feasibility is checked separately."
        target = attempt_dir / "alg_verifier.md"
        target.write_text("")
        result = client.edit_file(target, prompt, context=[attempt_dir / "solver.py"], log_dir=attempt_dir, step="llm_verifier")
        if not result.ok:
            return None, result.error or "Codex verifier failed"
        notes = target.read_text().strip()
        first = notes.splitlines()[0].strip().upper() if notes else ""
        return {"VERDICT: PASS": "pass", "VERDICT: FAIL": "fail"}.get(first), notes
