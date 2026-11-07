"""Generate and validate benchmark instances of the registered problem families (src.utils.families).

    python src/01_gen.py -s all -m validate -vl simple                  # validate the shipped instances in data/
    python src/01_gen.py -ir outputs/litgen_data -m validate -vl simple  # validate the small-scale instances
    python src/01_gen.py -ir /tmp/repro -p cvrp -m generate              # reproduce one family into a fresh root

Each family has one dataset root (its `dataset` in src.utils.families: data/, data_harder/, outputs/litgen_data/).
These roots are pinned by their campaign configs and are never written here: generation inside them only reuses
complete instances; a reproduction goes to a fresh --instances-root (--regenerate rebuilds instances there).
Prompts are rendered with the contract of the family's dataset; --prompt-contract-config re-renders the prompts of
an existing root under another config's contract.
"""

from __future__ import annotations

import argparse
import json
import math
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


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (ValueError, TypeError):
        return False


def _admitted_source(admission: str, source: Any) -> bool:
    from src.utils.dataset import ADMITTED_SOURCES

    return (source is None or isinstance(source, str)) and source in ADMITTED_SOURCES[admission]


def _verified_gold(path: Path) -> bool:
    try:
        gold = json.loads(path.read_text())
    except (OSError, ValueError):
        return False
    return isinstance(gold, dict) and not gold.get("error") and isinstance(gold.get("verification"), dict) and gold["verification"].get("ok") is True


def validate_instance(inst_dir: Path, validation_level: str = "strict") -> dict[str, Any]:
    """Validate one instance folder <root>/<suite>/<problem>/<instance> under the admission rule it was admitted with
    (src.utils.dataset.instance_admission: the rule of the pinned dataset root holding it, else the family's registered
    rule; src.utils.instance_suite defines the rules).

    simple: data.json, prompt.txt, gold_monolithic.json and meta.json exist and parse, and meta.json records a ground
      truth source the rule admits (none for the paper's own pipelines, "monolithic_mip" for "strict", either source
      for the reused families' "mip_or_certificate").
    strict: in addition, the instance is complete under that rule (src.utils.dataset.instance_objectives) and its
      objectives agree with it: an admitted MIP optimum matches gold_monolithic.json, a certificate matches the
      decomposition gold, a decomposition objective is not below the ground truth (beyond the family's tolerance), the
      stored golds pass the rule's own predicates (instance_suite.admission_issue under "strict": verified golds, MIP
      time and closed bounds, reference time, the 1 % gap rule and the family's certificate requirement; a closed
      certificate for a certificate-based ground truth under "mip_or_certificate"). A MIP-only or certificate-only
      admission of the reused families passes.
    """
    from src.utils.dataset import instance_admission, instance_objectives

    suite = _infer_suite(inst_dir)
    is_bd = suite == "bd"
    is_cg = suite == "cg"
    problem = Path(inst_dir).resolve().parent.name
    admission = instance_admission(inst_dir, problem) if suite in {"bd", "cg"} else None
    result: dict[str, Any] = {
        "name": inst_dir.name,
        "path": str(inst_dir),
        "suite": suite or "unknown",
        "problem": problem,
        "admission": admission,
        "has_data": (inst_dir / "data.json").exists(),
        "has_prompt": (inst_dir / "prompt.txt").exists(),
        "has_mono": (inst_dir / "gold_monolithic.json").exists(),
        "has_meta": (inst_dir / "meta.json").exists(),
        "has_bd": (inst_dir / "gold_bd.json").exists() if is_bd else False,
        "has_cg": (inst_dir / "gold_cg.json").exists() if is_cg else False,
        "mono_obj": None,
        "bd_obj": None,
        "cg_obj": None,
        "ground_truth_source": None,
        "reference_objective": None,
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

    meta = None
    if result["has_meta"]:
        try:
            meta = json.loads((inst_dir / "meta.json").read_text())
            if not isinstance(meta, dict):
                raise ValueError("not a JSON object")
            result["ground_truth_source"] = meta.get("ground_truth_source")
        except (OSError, ValueError) as exc:
            meta = None
            result["issues"].append(f"failed to parse meta.json: {exc}")

    def fail(status: str, issue: str) -> dict[str, Any]:
        result["status"] = status
        result["issues"].append(issue)
        return result

    if admission is None:
        return fail("unregistered_family", f"{problem!r} under suite {suite!r} is not a registered family (src.utils.families)")

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
            return fail("missing_required", "missing: " + ", ".join(missing))
        # A reference that exists but cannot be read is not a pass.
        if result["issues"]:
            result["status"] = "invalid_reference"
            return result
        if not _admitted_source(admission, result["ground_truth_source"]):
            return fail("not_admitted", f"meta.json ground_truth_source {result['ground_truth_source']!r} is not admitted by the {admission!r} rule")
        result["status"] = "pass_simple_test"
        return result

    decomposition_obj = result["cg_obj"] if is_cg else result["bd_obj"]
    has_decomposition = result["has_cg"] if is_cg else result["has_bd"]
    if meta is not None and not _admitted_source(admission, result["ground_truth_source"]):
        return fail("not_admitted", f"meta.json ground_truth_source {result['ground_truth_source']!r} is not admitted by the {admission!r} rule")
    objectives = instance_objectives(inst_dir, suite, problem, admission)
    if objectives is None:
        source = result["ground_truth_source"]
        if not result["has_mono"] or (source in (None, "monolithic_mip") and not _finite(result["mono_obj"])):
            return fail("missing_gold_mono", "missing gold_monolithic.json or objective")
        if not has_decomposition or (not _finite(decomposition_obj) and (source == "decomposition_certificate" or admission != "mip_or_certificate")):
            return fail(f"missing_gold_{suite}", f"missing gold_{suite}.json or objective")
        return fail("not_admitted", f"incomplete under the {admission!r} rule (data.json, prompt.txt, meta.json and both golds must be readable, with an admitted ground truth)")

    from src.utils.families import family
    from src.utils.instance_suite import GAP_RULE, admission_issue, certified

    truth, decomposed = objectives
    source = result["ground_truth_source"]
    if admission != "family":
        # The registered rules admit only verified golds (instance_suite: a gold with an error or a failed
        # verification is not eligible); the paper's own pipelines record no verification.
        mono_ok, decomposition_ok = (_verified_gold(inst_dir / name) for name in ("gold_monolithic.json", f"gold_{suite}.json"))
        if source == "monolithic_mip" and not mono_ok:
            return fail("not_admitted", "the admitted MIP optimum comes from an unverified or failed gold_monolithic.json")
        if source == "decomposition_certificate" and not decomposition_ok:
            return fail("not_admitted", f"the admitted certificate comes from an unverified or failed gold_{suite}.json")
        if not decomposition_ok:
            if admission == "strict":
                return fail("not_admitted", f"gold_{suite}.json failed or is not verified")
            decomposed = None  # the reused families' rule discards an unverified reference (MIP-only admission)
    result["reference_objective"] = truth
    truth, tol = float(truth), family(problem).tol
    scale = max(1.0, abs(truth))
    if source == "monolithic_mip" and not abs(float(result["mono_obj"]) - truth) <= tol * scale:
        return fail("objective_mismatch", f"meta.json ground truth {truth} differs from the monolithic objective {result['mono_obj']}")
    if source == "decomposition_certificate" and not abs(float(decomposed) - truth) <= tol * scale:
        return fail("objective_mismatch", f"meta.json ground truth {truth} differs from the certified decomposition objective {decomposed}")
    if admission != "family":
        # The registered predicates themselves (src.utils.instance_suite), applied to the stored golds.
        try:
            mono_gold, ref_gold = (json.loads((inst_dir / name).read_text()) for name in ("gold_monolithic.json", f"gold_{suite}.json"))
        except (OSError, ValueError) as exc:
            return fail("invalid_reference", f"failed to read the golds: {type(exc).__name__}: {exc}")
        if admission == "strict":
            try:
                reason, _ = admission_issue(family(problem), mono_gold, ref_gold)
            except (TypeError, ValueError) as exc:  # e.g. a non-numeric total_time: a bad record, not a batch abort
                return fail("invalid_reference", f"gold timing or bound fields are not numeric: {type(exc).__name__}: {exc}")
            if reason is not None:
                status = "not_admitted"
                if reason.startswith("decomposition gap"):
                    status = "objective_mismatch" if float(decomposed) < truth else "inadmissible_gap"
                return fail(status, f"the strict admission rule rejects it: {reason}")
        else:
            if source == "decomposition_certificate" and not certified(ref_gold, tol):
                return fail("not_admitted", f"ground truth from a decomposition certificate, but gold_{suite}.json is not certified")
            if decomposed is not None and certified(ref_gold, tol):
                # As _admit_mip_or_certificate: a certified reference must match the ground truth and any verified MIP optimum.
                if not abs(float(decomposed) - truth) <= tol * scale:
                    return fail("objective_mismatch", f"certified reference {decomposed} differs from the ground truth {truth}")
                if mono_ok and _finite(result["mono_obj"]) and not abs(float(result["mono_obj"]) - float(decomposed)) <= tol * max(1.0, abs(float(result["mono_obj"]))):
                    return fail("objective_mismatch", f"certified reference {decomposed} differs from the verified MIP optimum {result['mono_obj']}")
    if decomposed is None:  # the reused families' MIP-only admission
        result["status"] = "pass_simple_test"
        return result
    difference = float(decomposed) - truth
    # The paper rule keeps its historical absolute tolerance; the registered rules use the family's tolerance.
    if (abs(difference) < 1e-4) if admission == "family" else (abs(difference) <= tol * scale):
        result["status"] = "pass_simple_test"
    elif difference > 0:
        result["decomposition_gap"] = difference / scale
        if admission == "strict" and result["decomposition_gap"] > GAP_RULE:
            return fail("inadmissible_gap", f"decomposition gap {result['decomposition_gap']} above the {GAP_RULE} admission rule")
        result["status"] = "reference_gap"
    else:
        return fail("objective_mismatch", f"Objective mismatch: ground truth={truth}, decomposition={decomposed}")
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


def _validate_one(path: Path, validation_level: str) -> dict[str, Any]:
    try:
        return validate_instance(path, validation_level=validation_level)
    except (OSError, ValueError, TypeError) as exc:  # one unreadable instance never hides the others' results
        return {"name": path.name, "path": str(path), "status": "invalid_reference", "issues": [f"validation failed: {type(exc).__name__}: {exc}"]}


def validate_all(instances_root: Path, validation_level: str = "strict") -> list[dict[str, Any]]:
    return [_validate_one(p, validation_level) for p in _find_instance_dirs(instances_root)]


def print_summary(results: list[dict[str, Any]], root: Path) -> None:
    total = len(results)
    ok = [r for r in results if r["status"] == "pass_simple_test"]
    missing_required = [r for r in results if r["status"] == "missing_required"]
    missing_mono = [r for r in results if r["status"] == "missing_gold_mono"]
    missing_bd = [r for r in results if r["status"] == "missing_gold_bd"]
    missing_cg = [r for r in results if r["status"] == "missing_gold_cg"]
    mismatch = [r for r in results if r["status"] == "objective_mismatch"]
    reference_gaps = [r for r in results if r["status"] == "reference_gap"]
    not_admitted = [r for r in results if r["status"] in {"not_admitted", "inadmissible_gap", "unregistered_family", "invalid_reference"}]

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
    if not_admitted:
        print(f"  Not admitted:         {len(not_admitted)} (outside the registered admission rule)")
    print()

    for bucket, label in (
        (missing_required, "Missing required"),
        (missing_mono, "Missing gold mono"),
        (missing_bd, "Missing gold bd"),
        (missing_cg, "Missing gold cg"),
        (mismatch, "Objective mismatch"),
        (not_admitted, "Not admitted"),
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


def pinned_roots() -> set[Path]:
    """The family dataset roots (src.utils.families): pinned by their campaign configs, never written by this script."""
    from src.utils.families import dataset_roots

    return set(dataset_roots())


def selected_families(root: Path, suite: str, problem: str | None) -> list[str]:
    """The registered families to generate under root (in a pinned root: the families it holds)."""
    from src.utils.families import FAMILIES

    names = [name for name, entry in FAMILIES.items() if suite == "all" or entry.suite == suite]
    if root.resolve() in pinned_roots():
        names = [name for name in names if (root / FAMILIES[name].suite / name).is_dir()]
    if problem:
        if problem not in names:
            raise ValueError(f"Unknown problem '{problem}' for suite '{suite}' under {root}")
        names = [problem]
    return names


def generate(root: Path, args) -> None:
    from src.utils.dataset import instance_complete
    from src.utils.families import FAMILIES

    pinned = root.resolve() in pinned_roots()
    if pinned and args.regenerate:
        raise SystemExit(f"{root} is a pinned family dataset root; regenerate into a fresh --instances-root and compare")
    attempted = False
    try:
        for name in selected_families(root, args.suite, args.problem):
            entry = FAMILIES[name]
            directory = root / entry.suite / name
            missing = [f"{name}_{size}_seed{seed:03d}" for size in args.sizes for seed in range(args.seed_offset, args.seed_offset + args.num_instances)
                       if not instance_complete(directory / f"{name}_{size}_seed{seed:03d}", entry.suite, name)]
            if pinned and missing:
                raise SystemExit(f"{directory}: {missing} incomplete; a pinned dataset root is never written, generate into a fresh --instances-root")
            attempted |= bool(missing) or args.regenerate
            ensure_suite(entry.suite, name, directory, args.sizes, args.num_instances, args.seed_offset, args.regenerate)
    finally:
        # After any generation attempt, also a rejected or skipped (not admitted) one, so the manifest never
        # advertises an instance whose meta.json was removed.
        if attempted:
            write_manifest(root)


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
        help="strict checks completeness and objectives under the registered admission rule; simple checks data+prompt+mono+meta and the admitted ground-truth source only.",
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
    parser.add_argument("--regenerate", action="store_true", help="Explicitly regenerate existing instances (not inside a family dataset root).")
    parser.add_argument("--prompt-contract-config", type=Path, default=None,
                        help="Only re-render every prompt.txt under --instances-root with this campaign config's decomposition contract, then exit.")
    args = parser.parse_args()
    if args.prompt_contract_config is not None:
        from src.agent.prompts.snapshot import render_dataset_prompts
        from src.experiments.data_context import decomposition_contract_policy
        from src.experiments.plan import load_plan
        contract = decomposition_contract_policy(load_plan(args.prompt_contract_config)["prompt_protocol"])
        print(json.dumps(render_dataset_prompts(Path(args.instances_root), contract)))
        return

    root = Path(args.instances_root)

    if args.mode in {"generate", "both"}:
        generate(root, args)
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
