"""
Algorithm verifier for the agent. Responsible for verifying the algorithm of the agent.
"""

from __future__ import annotations

from aider.coders import Coder
from aider.io import InputOutput
from aider.models import Model

from pathlib import Path
from src.agent.files_manager import ArtifactManager

class AlgorithmVerifier:
    """LLM-based verifier for checking algorithmic correctness."""
    def run_llm_verifier(
        attempt_dir: Path,
        model: Model,
        auto_yes: bool,
        stream: bool,
        suite: str,
        lazy: bool,
    ) -> tuple[str | None, str]:
        """Run LLM-based algorithm verification and return verdict and notes."""
        instance_root = attempt_dir.parent.parent
        verifier_md = ArtifactManager.write_text(attempt_dir, "alg_verifier.md", "")
        solver_path = attempt_dir / "solver.py"
        data_path = instance_root / "data.json"
        prompt_path = instance_root / "prompt.txt"
        if suite == "bd" and lazy:
            prompt = (
                "You are the LLM verifier. Inspect solver.py and judge whether it actually implements "
                "Benders Decomposition using Gurobi lazy constraints rather than solving a monolithic MIP or a non-lazy approach.\n"
                "Return a short decision in alg_verifier.md.\n\n"
                "Format:\n"
                "VERDICT: PASS or FAIL\n"
                "NOTES: <1-3 bullet points>\n"
            )
        else:
            prompt = (
                "You are the LLM verifier. Inspect solver.py and judge whether it actually implements "
                f"{'Benders Decomposition' if suite == 'bd' else 'Column Generation'} rather than solving a trivial monolithic MIP.\n"
                "Return a short decision in alg_verifier.md.\n\n"
                "Format:\n"
                "VERDICT: PASS or FAIL\n"
                "NOTES: <1-3 bullet points>\n"
            )
        ArtifactManager.write_text(attempt_dir, "llm_verifier_prompt.md", prompt)
        coder = Coder.create(
            main_model=model,
            fnames=[str(verifier_md)],
            read_only_fnames=[str(solver_path.resolve()), str(data_path.resolve()), str(prompt_path.resolve())],
            io=InputOutput(yes=auto_yes),
            stream=stream,
            use_git=False,
            edit_format="diff",
        )
        coder.run(prompt)
        text = verifier_md.read_text().strip()
        verdict = None
        for line in text.splitlines():
            if line.strip().upper().startswith("VERDICT:"):
                verdict = line.split(":", 1)[1].strip().lower()
                break
        return verdict, text