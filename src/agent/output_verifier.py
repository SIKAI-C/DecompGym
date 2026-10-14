"""
Output verifier for the agent. Responsible for verifying the output of the agent.
"""

from __future__ import annotations

from pathlib import Path
import importlib.util
import json
import math
from typing import Any

from src.utils.json_utils import safe_load_json
from src.agent.solution_checks import check_bd_solution
from src.agent.output_contract import validate_output_contract


class VerificationError(RuntimeError):
    """Trusted verifier malfunction; stop for repair without rescoring or retrying."""

class OutputVerifier:
    """Verify solution outputs and compute status metrics."""

    @staticmethod
    def import_verifier(suite: str, problem: str):
        """Load the problem verifier module for a suite/problem."""
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
        """Run the problem verifier on an output file."""
        data = json.loads(data_path.read_text())
        try:
            output = json.loads(output_path.read_text())
        except (json.JSONDecodeError, UnicodeError) as exc:
            return {"ok": False, "issues": [f"invalid output JSON: {exc}"]}
        issues = validate_output_contract(suite, problem, data, output)
        if issues:
            return {"ok": False, "issues": issues, "failure_kind": "output_contract"}
        return OutputVerifier._verify_numerical_output(suite, problem, data_path, output_path, tol, data, output)

    @staticmethod
    def verify_reference_output(suite: str, problem: str, data_path: Path, output_path: Path, tol: float) -> dict[str, Any]:
        """Check a reference plan without attributing an agent method to historical gold."""
        data = json.loads(data_path.read_text())
        try:
            output = json.loads(output_path.read_text())
        except (json.JSONDecodeError, UnicodeError) as exc:
            return {"ok": False, "issues": [f"invalid output JSON: {exc}"]}
        if not isinstance(output, dict):
            return {"ok": False, "issues": ["reference output must be an object"]}
        return OutputVerifier._verify_numerical_output(suite, problem, data_path, output_path, tol, data, output)

    @staticmethod
    def _verify_numerical_output(suite, problem, data_path, output_path, tol, data, output):
        verifier, import_err = OutputVerifier.import_verifier(suite, problem)
        if import_err:
            raise VerificationError(import_err)
        if not hasattr(verifier, "verify_solution"):
            raise VerificationError("verify_solution not found")
        try:
            result = verifier.verify_solution(str(data_path), str(output_path), tol=tol)
            if not result.get("ok"):
                return result
            if suite == "bd":
                result = check_bd_solution(problem, data, output, tol)
            objective = result.get("objective_value")
            claimed, issues, _ = OutputVerifier.extract_objective(output)
            if claimed is None or objective is None or not math.isfinite(float(objective)):
                issues.append("a finite objective is required")
            elif abs(claimed - float(objective)) > tol * max(1.0, abs(float(objective))):
                issues.append(f"objective mismatch: claimed {claimed}, recomputed {objective}")
            result["issues"] = result.get("issues", []) + issues
            result["ok"] = bool(result.get("ok")) and not result["issues"]
            return result
        except Exception as exc:
            raise VerificationError(f"{suite}/{problem} verifier failed: {type(exc).__name__}: {exc}") from exc

    @staticmethod
    def load_gold_objective(inst_dir: Path, suite: str) -> float | None:
        """Load the gold objective from metadata or gold files."""
        meta_path = inst_dir / "meta.json"
        if meta_path.exists():
            meta = safe_load_json(meta_path)
            if meta is not None:
                val = meta.get("ground_truth_objective", meta.get("gold_objective"))
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
        """Extract objective value and message from output payload."""
        issues: list[str] = []
        if output is None:
            return None, issues, None
        objective = output.get("objective_value", output.get("best_objective"))
        if objective is not None:
            try:
                objective = float(objective)
                if not math.isfinite(objective):
                    raise ValueError("nonfinite objective")
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
        execution_timed_out: bool = False,
    ) -> tuple[str, float | None]:
        """Compute status and gap from execution/verifier results."""
        gap = None
        if objective is not None and gold_obj is not None:
            gap = abs(objective - gold_obj) / max(1.0, abs(gold_obj))

        gap_ok = gap is not None and ((gap_threshold is None) or (gap <= gap_threshold))

        if execution_timed_out:
            status = "timeout"
        elif not exec_ok or output is None:
            status = "exec_failed"
        elif objective is None:
            status = "objective_not_available"
        elif not verify_ok:
            status = "verify_failed"  # infeasible or failed verifier
        elif gold_obj is None:
            status = "reference_unavailable"
        elif not gap_ok:
            status = "feasible_not_optimal"  # feasible but not within gap threshold
        else:
            status = "ok"
        return status, gap
