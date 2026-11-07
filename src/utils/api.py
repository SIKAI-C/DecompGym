"""Small Python API for the registered problem families (src.utils.families) used by the command-line tools."""
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
from src.utils.families import FAMILIES, SIZES, dataset_roots, family
from src.utils.reference_output import reference_output


def reference_class_name(suite: str, problem: str, method: str) -> str:
    """"module.Class" of the family's monolithic MIP or decomposition reference (src.utils.families)."""
    return family(problem, suite).reference_class(method)


def generate_dataset(root: str | Path, *, problems: tuple[str, ...] | None = None, num_instances: int = 3, sizes: tuple[str, ...] = SIZES) -> list[dict[str, Any]]:
    """Generate the missing instances of the registered families (all by default) under root/<suite>/<problem>.

    root must be a fresh root: the family dataset roots are pinned by their campaign configs (src.utils.families)."""
    if Path(root).resolve() in dataset_roots():
        raise ValueError(f"{root} is a pinned family dataset root; generate into a fresh root and compare")
    records = []
    for problem in problems or tuple(FAMILIES):
        entry = family(problem)
        records.extend(ensure_suite(entry.suite, problem, Path(root) / entry.suite / problem, list(sizes), num_instances))
    write_manifest(Path(root))
    return records


def solve(suite: str, problem: str, data_path: str | Path, *, method: str = "decomposition", **solver_options: Any) -> dict[str, Any]:
    """Solve with a monolithic or decomposition reference implementation."""
    started = time.perf_counter()
    if method not in {"monolithic", "decomposition"}:
        raise ValueError("method must be monolithic or decomposition")
    name = reference_class_name(suite, problem, method)
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
    from src.experiments.verification import OutputVerifier

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
