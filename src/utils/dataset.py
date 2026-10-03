"""Generate only missing benchmark instances; existing complete data is reused."""
from __future__ import annotations

import importlib
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def instance_complete(directory: Path, suite: str) -> bool:
    required = ["data.json", "prompt.txt", "meta.json", "gold_monolithic.json", f"gold_{suite}.json"]
    for name in required:
        path = directory / name
        if not path.is_file():
            return False
        if path.suffix == ".json":
            try:
                payload = json.loads(path.read_text())
            except (ValueError, OSError):
                return False
            if name.startswith("gold_"):
                try:
                    if not math.isfinite(float(payload.get("best_objective"))):
                        return False
                except (ValueError, TypeError):
                    return False
    return True


def ensure_suite(suite: str, problem: str, root: Path, sizes: list[str], num_instances: int = 3, seed_offset: int = 0, regenerate: bool = False) -> list[dict[str, Any]]:
    if num_instances < 1:
        raise ValueError("num_instances must be positive")
    records = []
    module = None
    for size in sizes:
        for seed in range(seed_offset, seed_offset + num_instances):
            directory = root / f"{problem}_{size}_seed{seed:03d}"
            reused = not regenerate and instance_complete(directory, suite)
            if not reused:
                # Lazy import allows a cached dataset to be inspected without solver dependencies.
                module = module or importlib.import_module(f"src.{suite}.{problem}.c_gen")
                module.generate_suite(sizes=[size], num_instances=1, out_root=root, seed_offset=seed)
                meta_path = directory / "meta.json"
                if not meta_path.exists() and (directory / "gold_monolithic.json").exists() and (directory / f"gold_{suite}.json").exists():
                    mono = json.loads((directory / "gold_monolithic.json").read_text())
                    meta_path.write_text(json.dumps({"seed": seed, "size": size, "folder": directory.name, "ground_truth_objective": mono.get("best_objective")}, indent=2))
                if not instance_complete(directory, suite):
                    raise RuntimeError(f"Incomplete generated instance: {directory}; inspect the reference solver failure.")
            mono = json.loads((directory / "gold_monolithic.json").read_text())["best_objective"]
            decomposed = json.loads((directory / f"gold_{suite}.json").read_text())["best_objective"]
            records.append({"suite": suite, "problem": problem, "instance": directory.name, "reused": reused,
                            "reference_objective": mono, "decomposition_objective": decomposed,
                            "decomposition_gap": abs(decomposed - mono) / max(1.0, abs(mono))})
            print(f"{'Reused' if reused else 'Generated'} {directory}", flush=True)
    return records


def write_manifest(root: Path) -> dict[str, Any]:
    instances = []
    for data in sorted(root.glob("*/*/*/data.json")):
        suite, problem, name = data.relative_to(root).parts[:3]
        if suite not in {"bd", "cg"}:
            continue
        directory = data.parent
        record = {"suite": suite, "problem": problem, "instance": name,
                  "complete": instance_complete(directory, suite),
                  "data_sha256": hashlib.sha256(data.read_bytes()).hexdigest()}
        if record["complete"]:
            mono = json.loads((directory / "gold_monolithic.json").read_text())["best_objective"]
            decomposition = json.loads((directory / f"gold_{suite}.json").read_text())["best_objective"]
            record.update(reference_objective=mono, decomposition_objective=decomposition,
                          decomposition_gap=abs(decomposition - mono) / max(1.0, abs(mono)))
        instances.append(record)
    manifest = {"schema_version": 1, "num_instances": len(instances), "instances": instances}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
