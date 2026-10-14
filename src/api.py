"""Small Python API for the same problem families used by the command-line tools."""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
import importlib
import json
import math
from pathlib import Path
import tempfile
import time
from typing import Any

from src.utils.dataset import ensure_suite, write_manifest
from src.utils.reference_output import reference_output


PROBLEMS = {
    "bd": {
        "ufl": ("b_gold_monolithic.UFLBaseline", "b_gold_bd.UFLBenders"),
        "fctp": ("a_gold_monolithic.FCTPBaseline", "a_gold_bd.FCTPBenders"),
        "km": ("a_gold_monolithic.KMedianBaseline", "a_gold_bd.KMedianBenders"),
        "lrp": ("a_gold_monolithic.LRPBaseline", "a_gold_bd.LRPBenders"),
        "tsso": ("a_gold_monolithic.TSSOBaseline", "a_gold_bd.TSSOBenders"),
    },
    "cg": {
        "bp": ("a_gold_monolithic.BinPackingBaseline", "a_gold_cg.BinPackingColumnGeneration"),
        "csp": ("a_gold_monolithic.CuttingStockBaseline", "a_gold_cg.CuttingStockColumnGeneration"),
        "gap": ("a_gold_monolithic.GAPBaseline", "a_gold_cg.GAPColumnGeneration"),
        "shift": ("a_gold_monolithic.ShiftBaseline", "a_gold_cg.ShiftColumnGeneration"),
    },
}


def generate_dataset(root: str | Path = "data", *, suites: tuple[str, ...] = ("bd", "cg"), num_instances: int = 3, sizes: tuple[str, ...] = ("small", "medium", "large")) -> list[dict[str, Any]]:
    records = []
    for suite in suites:
        for problem in PROBLEMS[suite]:
            records.extend(ensure_suite(suite, problem, Path(root) / suite / problem, list(sizes), num_instances))
    write_manifest(Path(root))
    return records


def solve(suite: str, problem: str, data_path: str | Path, *, method: str = "decomposition", **solver_options: Any) -> dict[str, Any]:
    """Solve with a monolithic or decomposition reference implementation."""
    started = time.perf_counter()
    if method not in {"monolithic", "decomposition"}:
        raise ValueError("method must be monolithic or decomposition")
    name = PROBLEMS[suite][problem][0 if method == "monolithic" else 1]
    module_name, class_name = name.rsplit(".", 1)
    module = importlib.import_module(f"src.{suite}.{problem}.{module_name}")
    solver = getattr(module, class_name)(**solver_options)
    result = solver.run(str(data_path), started_at=started) if method == "decomposition" else solver.run(str(data_path))
    result = _jsonable(result)
    result["output"] = reference_output(suite, problem, json.loads(Path(data_path).read_text()), result)
    if method == "decomposition":
        result["total_time"] = time.perf_counter() - started
    return result


def verify(suite: str, problem: str, data_path: str | Path, output: dict[str, Any]) -> dict[str, Any]:
    """Apply the harness's numerical checks to a reference solution, including baselines."""
    from src.agent.output_verifier import OutputVerifier

    with tempfile.TemporaryDirectory(prefix="decompbench-verify-") as temporary:
        path = Path(temporary) / "output.json"
        path.write_text(json.dumps(output))
        return OutputVerifier.verify_reference_output(suite, problem, Path(data_path), path, 1e-6)


def _jsonable(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    return value
