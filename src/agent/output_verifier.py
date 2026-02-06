"""
Output verifier for the agent. Responsible for verifying the output of the agent.
"""

from __future__ import annotations

from pathlib import Path
import importlib.util
from typing import Any

from src.utils.json_utils import safe_load_json

class OutputVerifier:

    @staticmethod
    def import_verifier(suite: str, problem: str):
        verifier_path = Path(__file__).resolve().parent.parent / suite / problem / "d_verifier.py"
        if not verifier_path.exists():
            return None, f"verifier not found at {verifier_path}"

        module_name = f"{suite}_{problem}_verifier"
        spec = importlib.util.spec_from_file_location(module_name, verifier_path)
        if spec is None or spec.loader is None:
            return None, f"failed to load verifier from {verifier_path}"
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception as exc:
            return None, f"failed to import verifier '{verifier_path}': {exc}"
        return module, None


    @staticmethod
    def verify_output(suite: str, problem: str, data_path: Path, output_path: Path, tol: float) -> dict[str, Any]:
        verifier, import_err = OutputVerifier.import_verifier(suite, problem)
        if import_err:
            return {"ok": False, "issues": [import_err]}
        if not hasattr(verifier, "verify_solution"):
            return {"ok": False, "issues": ["verify_solution not found"]}
        return verifier.verify_solution(str(data_path), str(output_path), tol=tol)

    @staticmethod
    def load_gold_objective(inst_dir: Path, suite: str) -> float | None:
        meta_path = inst_dir / "meta.json"
        if meta_path.exists():
            meta = safe_load_json(meta_path)
            if meta is not None:
                val = meta.get("ground_truth_objective") or meta.get("gold_objective")
                if val is not None:
                    try:
                        return float(val)
                    except Exception:
                        pass

        gold_file = "gold_bd.json" if suite == "bd" else "gold_cg.json"
        for name in ["gold_monolithic.json", gold_file]:
            p = inst_dir / name
            if not p.exists():
                continue
            payload = safe_load_json(p)
            if payload is None:
                continue
            for key in ("best_objective", "objective_value"):
                if key in payload and payload[key] is not None:
                    try:
                        return float(payload[key])
                    except Exception:
                        continue
        return None

    @staticmethod
    def extract_objective(output: dict[str, Any] | None) -> tuple[float | None, list[str], str | None]:
        issues: list[str] = []
        if output is None:
            return None, issues, None
        objective = output.get("objective_value") or output.get("best_objective")
        if objective is not None:
            try:
                objective = float(objective)
            except Exception:
                issues.append("objective not numeric")
                objective = None
        output_msg = output.get("error_message") or output.get("message")
        return objective, issues, output_msg

    @staticmethod
    def assess_status(
        *,
        exec_ok: bool,
        output: dict[str, Any] | None,
        verify_ok: bool,
        objective: float | None,
        gold_obj: float | None,
        gap_threshold: float | None,
    ) -> tuple[str, float | None]:
        gap = None
        if objective is not None and gold_obj is not None:
            gap = abs(objective - gold_obj) / max(1.0, abs(gold_obj))

        gap_ok = (gap is None) or (gap_threshold is None) or (gap <= gap_threshold)

        if not exec_ok or output is None:
            status = "exec_failed"
        elif not verify_ok:
            status = "verify_failed"  # infeasible or failed verifier
        elif not gap_ok:
            status = "feasible_not_optimal"  # feasible but not within gap threshold
        else:
            status = "ok"
        return status, gap