"""Generate only missing benchmark instances; existing complete data is reused."""
from __future__ import annotations

import importlib
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def instance_objectives(directory: Path, suite: str, problem: str | None = None) -> tuple[Any, Any] | None:
    """(ground truth, decomposition objective or None) of a complete instance, or None if it is not complete.

    Complete: data.json, prompt.txt, meta.json and both golds are present and readable, and
      - without a recorded ground_truth_source (the paper pipelines, whose meta.json may be a fallback): both golds have
        a finite objective, and the ground truth is the monolithic one;
      - with one (src.utils.instance_suite): meta.json's ground_truth_objective is finite and is the ground truth; a
        "decomposition_certificate" source needs a finite decomposition objective, a "monolithic_mip" source a finite
        monolithic objective and, except under the reused families' historical rule ("mip_or_certificate"), a finite
        decomposition objective too."""
    payloads = {}
    for name in ("data.json", "prompt.txt", "meta.json", "gold_monolithic.json", f"gold_{suite}.json"):
        path = directory / name
        if not path.is_file():
            return None
        if path.suffix == ".json":
            try:
                payloads[name] = json.loads(path.read_text())
            except (ValueError, OSError):
                return None
    meta, golds = payloads["meta.json"], [payloads["gold_monolithic.json"], payloads[f"gold_{suite}.json"]]
    if not all(isinstance(value, dict) for value in (meta, *golds)):
        return None
    mono, decomposed = (gold.get("best_objective") for gold in golds)
    source = meta.get("ground_truth_source")
    if source is None:
        return (mono, decomposed) if _finite(mono) and _finite(decomposed) else None
    truth = meta.get("ground_truth_objective")
    historical = _admission(problem or directory.name.rsplit("_", 2)[0]) == "mip_or_certificate"
    needs = {"decomposition_certificate": (False, True), "monolithic_mip": (True, not historical)}.get(source)
    if needs is None or not _finite(truth) or (needs[0] and not _finite(mono)) or (needs[1] and not _finite(decomposed)):
        return None
    return truth, (decomposed if _finite(decomposed) else None)


def instance_complete(directory: Path, suite: str, problem: str | None = None) -> bool:
    return instance_objectives(directory, suite, problem) is not None


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (ValueError, TypeError):
        return False


def _admission(problem: str) -> str | None:
    """The registered admission of a family (src.utils.families); None for an unregistered name."""
    from src.utils.families import FAMILIES

    entry = FAMILIES.get(problem)
    return entry.admission if entry is not None else None


def _gap(truth: Any, decomposed: Any) -> float | None:
    return abs(decomposed - truth) / max(1.0, abs(truth)) if _finite(truth) and _finite(decomposed) else None


def ensure_suite(suite: str, problem: str, root: Path, sizes: list[str], num_instances: int = 3, seed_offset: int = 0, regenerate: bool = False) -> list[dict[str, Any]]:
    if num_instances < 1:
        raise ValueError("num_instances must be positive")
    records = []
    module = None
    for size in sizes:
        for seed in range(seed_offset, seed_offset + num_instances):
            directory = root / f"{problem}_{size}_seed{seed:03d}"
            reused = not regenerate and instance_complete(directory, suite, problem)
            if not reused:
                # Lazy import allows a cached dataset to be inspected without solver dependencies.
                module = module or importlib.import_module(f"src.{suite}.{problem}.c_gen")
                module.generate_suite(sizes=[size], num_instances=1, out_root=root, seed_offset=seed)
                meta_path = directory / "meta.json"
                if _admission(problem) in {"strict", "mip_or_certificate"}:
                    # src.utils.instance_suite owns admission: a missing meta.json is a skipped instance (the reused
                    # families' historical rule: no proved optimum and no certificate), never a synthesized one.
                    if not meta_path.exists():
                        print(f"Skipped {directory} (not admitted)", flush=True)
                        continue
                elif not meta_path.exists() and (directory / "gold_monolithic.json").exists() and (directory / f"gold_{suite}.json").exists():
                    mono = json.loads((directory / "gold_monolithic.json").read_text())
                    meta_path.write_text(json.dumps({"seed": seed, "size": size, "folder": directory.name, "ground_truth_objective": mono.get("best_objective")}, indent=2))
                if not instance_complete(directory, suite, problem):
                    raise RuntimeError(f"Incomplete generated instance: {directory}; inspect the reference solver failure.")
            truth, decomposed = instance_objectives(directory, suite, problem)
            records.append({"suite": suite, "problem": problem, "instance": directory.name, "reused": reused,
                            "reference_objective": truth, "decomposition_objective": decomposed,
                            "decomposition_gap": _gap(truth, decomposed)})
            print(f"{'Reused' if reused else 'Generated'} {directory}", flush=True)
    return records


def write_manifest(root: Path) -> dict[str, Any]:
    instances = []
    for data in sorted(root.glob("*/*/*/data.json")):
        suite, problem, name = data.relative_to(root).parts[:3]
        if suite not in {"bd", "cg"}:
            continue
        directory = data.parent
        objectives = instance_objectives(directory, suite, problem)
        record = {"suite": suite, "problem": problem, "instance": name,
                  "complete": objectives is not None,
                  "data_sha256": hashlib.sha256(data.read_bytes()).hexdigest()}
        if objectives is not None:
            truth, decomposition = objectives
            record.update(reference_objective=truth, decomposition_objective=decomposition,
                          decomposition_gap=_gap(truth, decomposition))
        instances.append(record)
    manifest = {"schema_version": 1, "num_instances": len(instances), "instances": instances}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
