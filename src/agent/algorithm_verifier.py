"""Review decomposition implementation through a fresh Codex session."""
from pathlib import Path

from src.agent.codex_client import CodexClient
from src.agent.decomposition_policy import METHOD_NAMES, method_requirements


def algorithm_review_prompt(suite: str) -> str:
    return "\n\n".join((
        f"Inspect the supplied solver.py and determine whether it implements {METHOD_NAMES[suite]} and substantively uses that method to produce the reported solution. Apply the following method contract, which is also supplied to the solver author:",
        method_requirements(suite),
        "Trace the entry point, relevant reachable call paths, information passed back to the master, and the code that writes output.json. Method names, comments, unused decomposition functions, or loops that do not feed subproblem information back into the master are insufficient. For an initialization or final solve, identify which decisions are fixed, which variables or columns are optimized, and where its result is used; an LP or MIP call by itself is not a violation.",
        "Return FAIL if a reachable solution-producing path uses a prohibited separate full-original-model solve, even if decomposition also runs. For conditional fallbacks, cite the triggering conditions and role from source without assuming they ran. Unused helpers or provably unreachable branches alone are not a reason for FAIL. A genuine decomposition can converge or stop without adding a cut or column in a particular instance, and may return its initialized feasible incumbent.",
        "This is an algorithm-use assessment, not a full mathematical proof of every cut, bound, or pricing formulation. Execution, feasibility, output consistency, and objective quality are checked separately. An incorrect numerical result alone is not a reason for algorithm FAIL; determine whether the requested decomposition is substantively implemented and used. Do not require a particular cut construction, initialization heuristic, pricing algorithm, or number of iterations.",
        "Do not execute or modify solver.py. Do not claim to have observed runtime behavior when reviewing source only. Write alg_verifier.md with exactly a first line VERDICT: PASS or VERDICT: FAIL, followed by NOTES: and concise code evidence supporting the verdict.",
    ))


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
