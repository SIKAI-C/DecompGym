"""
Unified validator for benchmark instances across all problem types.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


def _load_objective(path: Path) -> float | None:
    try:
        payload = json.loads(path.read_text())
        return payload.get("best_objective") or payload.get("objective_value")
    except Exception as exc:  # pragma: no cover - simple utility
        raise ValueError(f"failed to parse {path.name}: {exc}") from exc


def validate_instance(inst_dir: Path) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "name": inst_dir.name,
        "path": str(inst_dir),
        "has_data": (inst_dir / "data.json").exists(),
        "has_prompt": (inst_dir / "prompt.txt").exists(),
        "has_mono": (inst_dir / "gold_monolithic.json").exists(),
        "has_bd": (inst_dir / "gold_bd.json").exists(),
        "mono_obj": None,
        "bd_obj": None,
        "status": "unknown",
        "issues": [],
    }

    if not result["has_data"]:
        result["issues"].append("missing data.json")
    if not result["has_prompt"]:
        result["issues"].append("missing prompt.txt")

    if result["has_mono"]:
        try:
            result["mono_obj"] = _load_objective(inst_dir / "gold_monolithic.json")
        except ValueError as exc:
            result["issues"].append(str(exc))

    if result["has_bd"]:
        try:
            result["bd_obj"] = _load_objective(inst_dir / "gold_bd.json")
        except ValueError as exc:
            result["issues"].append(str(exc))

    if not result["has_mono"] or not result["has_bd"]:
        result["status"] = "missing_ground_truth"
        result["issues"].append("ground truth incomplete")
    elif result["mono_obj"] is None or result["bd_obj"] is None:
        result["status"] = "invalid_objectives"
        result["issues"].append("objective values missing from JSON")
    elif abs(result["mono_obj"] - result["bd_obj"]) < 1e-4:
        result["status"] = "ok"
    else:
        result["status"] = "objective_mismatch"
        result["issues"].append(
            f"Objective mismatch: mono={result['mono_obj']}, bd={result['bd_obj']}"
        )

    return result


def _find_instance_dirs(root: Path) -> List[Path]:
    if not root.exists():
        return []

    dirs: Set[Path] = set()
    if (root / "data.json").exists():
        dirs.add(root)

    for data_file in root.rglob("data.json"):
        dirs.add(data_file.parent)

    return sorted(dirs)


def validate_all(instances_root: Path) -> List[Dict[str, Any]]:
    return [validate_instance(p) for p in _find_instance_dirs(instances_root)]


def print_summary(results: List[Dict[str, Any]], root: Path) -> None:
    total = len(results)
    ok = [r for r in results if r["status"] == "ok"]
    missing = [r for r in results if r["status"] == "missing_ground_truth"]
    mismatch = [r for r in results if r["status"] == "objective_mismatch"]
    invalid = [r for r in results if r["status"] == "invalid_objectives"]

    print(f"\n{'='*60}")
    print(f"VALIDATION SUMMARY for {root}")
    print(f"{'='*60}")
    print(f"Total instances:        {total}")
    print(f"  OK:                   {len(ok)}")
    print(f"  Missing ground truth: {len(missing)}")
    print(f"  Objective mismatch:   {len(mismatch)}")
    print(f"  Invalid objectives:   {len(invalid)}\n")

    for bucket, label in ((missing, "Missing ground truth"), (mismatch, "Objective mismatch")):
        if bucket:
            print(f"{label}:")
            for r in bucket:
                print(f"  - {r['name']}")
            print()


def remove_instances(
    instances_root: Path,
    results: List[Dict[str, Any]],
    remove_missing: bool = False,
    remove_mismatch: bool = False,
    pattern: Optional[str] = None,
) -> int:
    removed = 0
    for r in results:
        should_remove = False
        reason = ""

        if pattern and pattern in r["name"]:
            should_remove = True
            reason = f"matches pattern '{pattern}'"
        elif remove_missing and r["status"] == "missing_ground_truth":
            should_remove = True
            reason = "missing ground truth"
        elif remove_mismatch and r["status"] == "objective_mismatch":
            should_remove = True
            reason = "objective mismatch"

        if should_remove:
            inst_path = Path(r["path"])
            if inst_path.exists():
                print(f"Removing {r['name']} ({reason})")
                shutil.rmtree(inst_path)
                removed += 1

    # Refresh results if we removed anything
    return removed


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate benchmark instance folders.")
    parser.add_argument(
        "--instances-root",
        type=str,
        default="./z_artifacts/benchmarks",
        help="Root directory containing instance folders (problem-specific or all problems).",
    )
    parser.add_argument(
        "--problem",
        type=str,
        default=None,
        help="Optional problem name to append to instances-root (e.g., 'ufl', 'tsso').",
    )
    parser.add_argument("--remove-missing", action="store_true", help="Remove instances missing ground truth files.")
    parser.add_argument("--remove-mismatch", action="store_true", help="Remove instances where mono != bd.")
    parser.add_argument("--remove-bad", action="store_true", help="Remove all problematic instances.")
    parser.add_argument("--remove-pattern", type=str, default=None, help="Remove instances whose name contains string.")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of a human summary.")
    args = parser.parse_args()

    root = Path(args.instances_root)
    if args.problem:
        root = root / args.problem

    if not root.exists():
        print(f"Error: {root} does not exist")
        return

    results = validate_all(root)
    if not results:
        print(f"No instances found in {root}")
        return

    if args.json:
        print(json.dumps(results, indent=2))
        return

    print_summary(results, root)

    remove_missing = args.remove_missing or args.remove_bad
    remove_mismatch = args.remove_mismatch or args.remove_bad
    if remove_missing or remove_mismatch or args.remove_pattern:
        removed = remove_instances(
            root,
            results,
            remove_missing=remove_missing,
            remove_mismatch=remove_mismatch,
            pattern=args.remove_pattern,
        )
        print(f"\nRemoved {removed} instances.")


if __name__ == "__main__":
    main()
