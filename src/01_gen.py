"""Generate and validate the 81 benchmark instances in data/ (problem families in src/bd and src/cg).

    python src/01_gen.py -s all -m validate -vl simple   # validate the shipped instances
    python src/01_gen.py -s all -m both -vl strict       # generate missing instances, then validate with all references

Existing instances are reused; pass --regenerate to rebuild them.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.dataset import ensure_suite, write_manifest  # noqa: E402

os.environ.setdefault("MPLBACKEND", "Agg")


def _load_objective(path: Path) -> float | None:
    try:
        payload = json.loads(path.read_text())
        return payload.get("best_objective", payload.get("objective_value"))
    except Exception as exc:  # pragma: no cover - simple utility
        raise ValueError(f"failed to parse {path.name}: {exc}") from exc


def _infer_suite(inst_dir: Path) -> str | None:
    for parent in inst_dir.resolve().parents:
        if parent.name in {"bd", "cg"}:
            return parent.name
    return None


def validate_instance(inst_dir: Path, validation_level: str = "strict") -> dict[str, Any]:
    suite = _infer_suite(inst_dir)
    is_bd = suite == "bd"
    is_cg = suite == "cg"
    result: dict[str, Any] = {
        "name": inst_dir.name,
        "path": str(inst_dir),
        "suite": suite or "unknown",
        "has_data": (inst_dir / "data.json").exists(),
        "has_prompt": (inst_dir / "prompt.txt").exists(),
        "has_mono": (inst_dir / "gold_monolithic.json").exists(),
        "has_meta": (inst_dir / "meta.json").exists(),
        "has_bd": (inst_dir / "gold_bd.json").exists() if is_bd else False,
        "has_cg": (inst_dir / "gold_cg.json").exists() if is_cg else False,
        "mono_obj": None,
        "bd_obj": None,
        "cg_obj": None,
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

    if result["has_cg"]:
        try:
            result["cg_obj"] = _load_objective(inst_dir / "gold_cg.json")
        except ValueError as exc:
            result["issues"].append(str(exc))

    if validation_level == "simple":
        missing = []
        if not result["has_data"]:
            missing.append("data.json")
        if not result["has_prompt"]:
            missing.append("prompt.txt")
        if not result["has_mono"]:
            missing.append("gold_monolithic.json")
        if not result["has_meta"]:
            missing.append("meta.json")
        if missing:
            result["status"] = "missing_required"
            result["issues"].append("missing: " + ", ".join(missing))
            return result
        # A reference that exists but cannot be read is not a pass.
        result["status"] = "invalid_reference" if result["issues"] else "pass_simple_test"
        return result

    if not result["has_mono"] or result["mono_obj"] is None:
        result["status"] = "missing_gold_mono"
        result["issues"].append("missing gold_monolithic.json or objective")
    elif is_bd and (not result["has_bd"] or result["bd_obj"] is None):
        result["status"] = "missing_gold_bd"
        result["issues"].append("missing gold_bd.json or objective")
    elif is_cg and (not result["has_cg"] or result["cg_obj"] is None):
        result["status"] = "missing_gold_cg"
        result["issues"].append("missing gold_cg.json or objective")
    elif abs(result["mono_obj"] - (result["cg_obj"] if is_cg else result["bd_obj"])) < 1e-4:
        result["status"] = "pass_simple_test"
    elif (result["cg_obj"] if is_cg else result["bd_obj"]) > result["mono_obj"]:
        result["status"] = "reference_gap"
        result["decomposition_gap"] = ((result["cg_obj"] if is_cg else result["bd_obj"]) - result["mono_obj"]) / max(1.0, abs(result["mono_obj"]))
    else:
        result["status"] = "objective_mismatch"
        result["issues"].append(f"Objective mismatch: mono={result['mono_obj']}, decomposition={result['cg_obj'] if is_cg else result['bd_obj']}")

    return result


def _find_instance_dirs(root: Path) -> list[Path]:
    if not root.exists():
        return []

    dirs: set[Path] = set()
    if (root / "data.json").exists():
        dirs.add(root)

    for data_file in root.rglob("data.json"):
        dirs.add(data_file.parent)

    return sorted(dirs)


def validate_all(instances_root: Path, validation_level: str = "strict") -> list[dict[str, Any]]:
    return [validate_instance(p, validation_level=validation_level) for p in _find_instance_dirs(instances_root)]


def print_summary(results: list[dict[str, Any]], root: Path) -> None:
    total = len(results)
    ok = [r for r in results if r["status"] == "pass_simple_test"]
    missing_required = [r for r in results if r["status"] == "missing_required"]
    missing_mono = [r for r in results if r["status"] == "missing_gold_mono"]
    missing_bd = [r for r in results if r["status"] == "missing_gold_bd"]
    missing_cg = [r for r in results if r["status"] == "missing_gold_cg"]
    mismatch = [r for r in results if r["status"] == "objective_mismatch"]
    reference_gaps = [r for r in results if r["status"] == "reference_gap"]

    print(f"\n{'=' * 60}")
    print(f"VALIDATION SUMMARY for {root}")
    print(f"{'=' * 60}")
    print(f"Total instances:        {total}")
    print(f"  PASS_SIMPLE TEST:     {len(ok)}")
    if missing_required:
        print(f"  Missing required:     {len(missing_required)}")
    print(f"  Missing gold mono:    {len(missing_mono)}")
    print(f"  Missing gold bd:      {len(missing_bd)}")
    print(f"  Missing gold cg:      {len(missing_cg)}")
    print(f"  Objective mismatch:   {len(mismatch)}")
    print(f"  Decomposition gaps:   {len(reference_gaps)} (retained as educational examples)")
    print()

    for bucket, label in (
        (missing_required, "Missing required"),
        (missing_mono, "Missing gold mono"),
        (missing_bd, "Missing gold bd"),
        (missing_cg, "Missing gold cg"),
        (mismatch, "Objective mismatch"),
    ):
        if bucket:
            print(f"{label}:")
            for r in bucket:
                print(f"  - {r['name']}")
            print()


def remove_instances(
    instances_root: Path,
    results: list[dict[str, Any]],
    remove_missing: bool = False,
    remove_mismatch: bool = False,
    remove_missing_mono: bool = False,
    remove_missing_bd: bool = False,
    remove_missing_cg: bool = False,
    pattern: str | None = None,
) -> int:
    removed = 0
    for r in results:
        should_remove = False
        reason = ""

        if pattern and pattern in r["name"]:
            should_remove = True
            reason = f"matches pattern '{pattern}'"
        elif remove_missing and r["status"] in {"missing_gold_mono", "missing_gold_bd", "missing_gold_cg"}:
            should_remove = True
            reason = "missing ground truth"
        elif remove_missing_mono and r["status"] == "missing_gold_mono":
            should_remove = True
            reason = "missing gold mono"
        elif remove_missing_bd and r["status"] == "missing_gold_bd":
            should_remove = True
            reason = "missing gold bd"
        elif remove_missing_cg and r["status"] == "missing_gold_cg":
            should_remove = True
            reason = "missing gold cg"
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
    parser = argparse.ArgumentParser(description="Generate and/or validate benchmark instances.")
    parser.add_argument(
        "--instances-root",
        "-ir",
        type=str,
        default="data",
        help="Root directory containing bd/ and cg/ benchmarks.",
    )
    parser.add_argument(
        "--problem",
        "-p",
        type=str,
        default=None,
        help="Optional problem name under the suite (e.g., 'ufl', 'gap').",
    )
    parser.add_argument(
        "--suite",
        "-s",
        type=str,
        default="all",
        choices=["bd", "cg", "all"],
        help="Benchmark suite to operate on.",
    )
    parser.add_argument(
        "--mode",
        "-m",
        type=str,
        default="both",
        choices=["generate", "validate", "both"],
        help="Run generation, validation, or both.",
    )
    parser.add_argument(
        "--validation-level",
        "-vl",
        type=str,
        default="strict",
        choices=["strict", "simple"],
        help="strict checks mono+bd/cg; simple checks data+prompt+mono+meta only.",
    )
    parser.add_argument(
        "--num-instances",
        "-ni",
        type=int,
        default=3,
        help="Number of instances per size when generating.",
    )
    parser.add_argument(
        "--seed-offset",
        "-so",
        type=int,
        default=0,
        help="Seed offset for generation.",
    )
    parser.add_argument(
        "--remove-missing",
        "-rm",
        action="store_true",
        help="Remove instances missing any gold file.",
    )
    parser.add_argument(
        "--remove-missing-mono",
        action="store_true",
        help="Remove instances missing gold monolithic.",
    )
    parser.add_argument(
        "--remove-missing-bd",
        action="store_true",
        help="Remove instances missing gold bd.",
    )
    parser.add_argument(
        "--remove-missing-cg",
        action="store_true",
        help="Remove instances missing gold cg.",
    )
    parser.add_argument(
        "--remove-mismatch",
        "-rmm",
        action="store_true",
        help="Remove instances where mono != bd/cg.",
    )
    parser.add_argument(
        "--remove-bad",
        "-rb",
        action="store_true",
        help="Remove all problematic instances.",
    )
    parser.add_argument(
        "--remove-pattern",
        "-rp",
        type=str,
        default=None,
        help="Remove instances whose name contains string.",
    )
    parser.add_argument(
        "--json",
        "-j",
        action="store_true",
        help="Print JSON instead of a human summary.",
    )
    parser.add_argument("--sizes", nargs="+", choices=["small", "medium", "large"], default=["small", "medium", "large"])
    parser.add_argument("--regenerate", action="store_true", help="Explicitly regenerate existing instances.")
    args = parser.parse_args()

    root = Path(args.instances_root)

    if args.mode in {"generate", "both"}:
        sizes = args.sizes
        suite_map = {
            "bd": ["fctp", "km", "lrp", "tsso", "ufl"],
            "cg": ["bp", "csp", "gap", "shift"],
        }
        suites = ["bd", "cg"] if args.suite == "all" else [args.suite]
        for suite in suites:
            problems = suite_map[suite]
            if args.problem:
                if args.problem not in problems:
                    raise ValueError(f"Unknown problem '{args.problem}' for suite '{suite}'")
                problems = [args.problem]
            for prob in problems:
                out_root = root / suite / prob
                ensure_suite(suite, prob, out_root, sizes, args.num_instances, args.seed_offset, args.regenerate)
        write_manifest(root)

    if args.mode == "generate":
        return

    validate_root = root
    if args.suite in {"bd", "cg"}:
        validate_root = validate_root / args.suite
    if args.problem:
        validate_root = validate_root / args.problem

    if not validate_root.exists():
        raise SystemExit(f"Error: {validate_root} does not exist")

    results = validate_all(validate_root, validation_level=args.validation_level)
    if not results:
        raise SystemExit(f"No instances found in {validate_root}")

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print_summary(results, validate_root)
    # Positive decomposition gaps are measured reference outcomes, not failures.
    failed = [r for r in results if r["status"] not in ("pass_simple_test", "reference_gap")]

    remove_missing = args.remove_missing or args.remove_bad
    remove_mismatch = args.remove_mismatch or args.remove_bad
    if (
        remove_missing
        or remove_mismatch
        or args.remove_missing_mono
        or args.remove_missing_bd
        or args.remove_missing_cg
        or args.remove_pattern
    ):
        removed = remove_instances(
            validate_root,
            results,
            remove_missing=remove_missing,
            remove_mismatch=remove_mismatch,
            remove_missing_mono=args.remove_missing_mono,
            remove_missing_bd=args.remove_missing_bd,
            remove_missing_cg=args.remove_missing_cg,
            pattern=args.remove_pattern,
        )
        print(f"\nRemoved {removed} instances.")
    if failed:
        raise SystemExit(f"{len(failed)} instance(s) failed validation: " + ", ".join(r["name"] for r in failed[:10]))


if __name__ == "__main__":
    main()
