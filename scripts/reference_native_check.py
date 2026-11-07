"""Run each saved reference decomposition through the agents' native sandbox (60 s, 1 thread) and score it like a campaign.

Restored from the paper's scripts/reference_calibration.py (DecompBench c6619a5, formal-v2 variant) and adapted to the
cleaned layout and the literature-recipe small-scale dataset; see `--help`.
"""
from __future__ import annotations

import argparse
import ast
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
DATASET = REPO / "outputs/litgen_data"
from src.experiments.sandbox import NativeSandboxExecutor
from src.experiments.verification import OutputVerifier
from src.experiments.docker_setup import license_file


GAP_THRESHOLD = 0.01  # the campaign's 1 % rule (paired_runner gap_threshold)

# Formal decomposition reference per family, as src.utils.api dispatches method="decomposition", with the generators'
# reference options (src/<suite>/<fam>/c_gen.py; time_limit=60 is passed separately, as the generators do).
REFERENCES = {
    "ufl": ("bd", "b_gold_bd", "UFLBenders", {}),
    "km": ("bd", "a_gold_bd", "KMedianBenders", {}),
    "fctp": ("bd", "a_gold_bd", "FCTPBenders", {}),
    "scfl": ("bd", "a_gold_bd", "SCFLBenders", {}),
    "lbbd_ps": ("bd", "a_gold_bd", "PSBenders", {}),
    "bp": ("cg", "a_gold_cg", "BinPackingColumnGeneration", {"ffd_warm_start": True}),
    "csp": ("cg", "a_gold_cg", "CuttingStockColumnGeneration", {}),
    "gap": ("cg", "a_gold_cg", "GAPColumnGeneration", {}),
    "bppc": ("cg", "a_gold_cg", "BPPCBranchAndPrice", {}),
    "clrp": ("cg", "a_gold_cg", "CLRPColumnGeneration", {}),
    "cvrp": ("cg", "a_gold_cg", "CVRPColumnGeneration", {}),
    "vrptw": ("cg", "a_gold_cg", "BranchAndPrice", {}),
    "pdptw": ("cg", "a_gold_cg", "PDPTWBranchAndPrice", {}),
    "rcsp": ("cg", "a_gold_cg", "RCSPColumnGeneration", {}),
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


class RemoveUnusedPlots(ast.NodeTransformer):
    def visit_Import(self, node):
        node.names = [alias for alias in node.names if not alias.name.startswith("matplotlib")]
        return node if node.names else None

    def visit_ImportFrom(self, node):
        return None if (node.module or "").startswith("matplotlib") else node

    def visit_FunctionDef(self, node):
        if node.name in {"plot_bounds", "plot_instance", "plot_solution", "_ensure_ax", "_ensure_fig_ax"}:
            return None
        return self.generic_visit(node)


def module_path(name):
    path = REPO / (name.replace(".", "/") + ".py")
    return path if path.is_file() else None


def src_imports(name):
    """Every src.* module named by any import statement in the module (module level or inside functions)."""
    found = set()
    for node in ast.walk(ast.parse(module_path(name).read_text())):
        if isinstance(node, ast.Import):
            if any(alias.name.startswith("src.") for alias in node.names):
                raise RuntimeError(f"{name}: 'import src.x' is not supported by the in-memory loader")
        elif isinstance(node, ast.ImportFrom) and (node.level or (node.module or "").split(".")[0] == "src"):
            if node.level:
                raise RuntimeError(f"{name}: relative imports are not supported by the in-memory loader")
            for alias in node.names:
                found.add(node.module + "." + alias.name if module_path(node.module + "." + alias.name) else node.module)
    missing = sorted(item for item in found if module_path(item) is None)
    if missing:
        raise RuntimeError(f"{name}: imports {missing} that are not source modules")
    return found


def import_closure(roots):
    """The real src.* import closure of the roots, dependencies first (the loader executes modules in this order)."""
    order, state = [], {}
    def visit(name):
        if state.get(name) == "done":
            return
        if state.get(name) == "active":
            raise RuntimeError(f"import cycle through {name}")
        state[name] = "active"
        for dependency in sorted(src_imports(name)):
            visit(dependency)
        state[name] = "done"
        order.append(name)
    for root in roots:
        visit(root)
    return order


def bundle_sources(problem):
    suite, module, cls, _ = REFERENCES[problem]
    names = import_closure([f"src.{suite}.{problem}.{module}", "src.utils.reference_output"])
    sources, inventory = {}, {}
    for name in names:
        path = module_path(name)
        original = path.read_text()
        tree = RemoveUnusedPlots().visit(ast.parse(original))
        ast.fix_missing_locations(tree)
        sources[name] = ast.unparse(tree) + "\n"
        inventory[str(path.relative_to(REPO))] = digest(path)
    return sources, inventory, f"src.{suite}.{problem}.{module}", cls


def jsonable_source():
    """src.utils.api._jsonable verbatim: api.solve applies it to the reference result before reference_output."""
    path = REPO / "src/utils/api.py"
    text = path.read_text()
    node = next(item for item in ast.parse(text).body if isinstance(item, ast.FunctionDef) and item.name == "_jsonable")
    return ast.get_source_segment(text, node), digest(path)


def solver_source(problem, variant):
    sources, inventory, module, cls = bundle_sources(problem)
    suite, options = REFERENCES[problem][0], REFERENCES[problem][3]
    jsonable, api_digest = jsonable_source()
    inventory["src/utils/api.py#_jsonable"] = api_digest
    source = f'''from __future__ import annotations
import time
STARTED = time.perf_counter()
import json, sys, types, math
from dataclasses import asdict, is_dataclass
from typing import Any
from pathlib import Path
import gurobipy as gp
from gurobipy import GRB
SOURCES = {sources!r}
VARIANT = {variant!r}
PROBLEM = {problem!r}
SUITE = {suite!r}
OPTIONS = {options!r}
MODEL_SOLVES = []
OriginalModel = gp.Model
class ObservedModel(OriginalModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.Params.OutputFlag = 0
    def optimize(self, *args, **kwargs):
        started = time.monotonic()
        super().optimize(*args, **kwargs)
        MODEL_SOLVES.append({{"name": self.ModelName, "threads": self.Params.Threads, "seconds": time.monotonic() - started, "status": self.Status, "variables": self.NumVars, "constraints": self.NumConstrs}})
gp.Model = ObservedModel
for name in SOURCES:
    pieces = name.split('.')
    for end in range(1, len(pieces)):
        parent = '.'.join(pieces[:end])
        if parent not in sys.modules:
            package = types.ModuleType(parent)
            package.__path__ = []
            sys.modules[parent] = package
    module = types.ModuleType(name)
    module.__file__ = '<bundled-reference/' + name + '>'
    sys.modules[name] = module
    exec(compile(SOURCES[name], module.__file__, 'exec'), module.__dict__)
from src.utils.reference_output import reference_output
ReferenceClass = sys.modules[{module!r}].__dict__[{cls!r}]
DATA = json.loads(Path('../../data.json').read_text())
{jsonable}
def publish(result, termination='reference_return'):
    result = _jsonable(result)
    value = reference_output(SUITE, PROBLEM, DATA, result)
    value.setdefault('status', 'feasible')
    value.setdefault('method', 'benders_decomposition' if SUITE == 'bd' else 'column_generation')
    value['_calibration'] = {{'variant': VARIANT, 'termination': termination, 'whole_script_seconds': time.perf_counter() - STARTED, 'reported_loop_seconds': result.get('total_time'), 'iterations': result.get('iters'), 'model_solves': MODEL_SOLVES, 'uses_numpy': True, 'global_lb': result.get('global_lb') if isinstance(result.get('global_lb'), (int, float)) and math.isfinite(result['global_lb']) else None}}
    value['_calibration']['reference_result'] = {{key: result.get(key) for key in ('reference_policy', 'termination_reason', 'relative_gap', 'target_gap', 'target_gap_met', 'proven_optimal', 'optimality_tolerance', 'status', 'threads', 'time_limit', 'model_solves', 'phase_one_records')}}
    Path('output.json').write_text(json.dumps(value, allow_nan=False, default=lambda value: value.item() if hasattr(value,'item') else value.tolist()))
instance = ReferenceClass(time_limit=60.0, **OPTIONS)
publish(instance.run('../../data.json', started_at=STARTED), 'formal_reference_return')
'''
    return source, inventory


def instances(problems, selected):
    """The dataset inventory: every family needs its 9 instances (3 sizes x seeds 0-2); every requested name must exist."""
    found = []
    for problem in problems:
        suite = REFERENCES[problem][0]
        names = sorted(path.parent.name for path in (DATASET / suite / problem).glob("*/data.json"))
        expected = sorted(f"{problem}_{size}_seed{seed:03d}" for size in ("small", "medium", "large") for seed in range(3))
        if names != expected:
            raise RuntimeError(f"{problem}: dataset inventory {names} differs from the expected {expected}")
        found += [(problem, name) for name in names if not selected or name in selected]
    if selected and sorted({name for _, name in found}) != sorted(set(selected)):
        raise RuntimeError(f"requested instances not found: {sorted(set(selected) - {name for _, name in found})}")
    return found


def scoring_inputs(source):
    """Files that decide the score besides data.json (gold metadata); a cached record is reused only if they are unchanged."""
    return {name: digest(source / name) for name in ("meta.json", "gold_monolithic.json") if (source / name).exists()}


def run_case(executor, destination, problem, instance, variant):
    suite = REFERENCES[problem][0]
    source = DATASET / suite / problem / instance
    case = destination / variant / suite / problem / instance
    result_path = case / "calibration.json"
    solver, inventory = solver_source(problem, variant)
    if result_path.exists():
        previous = json.loads(result_path.read_text())
        previous_inventory = json.loads((case / "source_inventory.json").read_text())
        expected_solver = hashlib.sha256(solver.encode()).hexdigest()
        if previous_inventory["original_sources"] != inventory or previous["input_sha256"] != digest(source / "data.json") or previous["solver_sha256"] != expected_solver or previous.get("scoring_inputs_sha256") != scoring_inputs(source) or digest(case / "attempts/attempt_01/solver.py") != expected_solver:
            raise RuntimeError("Calibration inputs/source changed; use a new variant or output directory and preserve the old evidence")
        if previous.get("output_sha256") and digest(case / "attempts/attempt_01/output.json") != previous["output_sha256"]:
            raise RuntimeError("Saved calibration output changed")
        return previous
    attempt = case / "attempts/attempt_01"
    attempt.mkdir(parents=True)
    shutil.copyfile(source / "data.json", case / "data.json")
    (attempt / "solver.py").write_text(solver)
    save(case / "source_inventory.json", {"original_sources": inventory, "bundled_modules": [name for name in inventory if not name.endswith("#_jsonable")], "reference_options": {"time_limit": 60.0, **REFERENCES[problem][3]},
                                          "adapter": "Inline formal reference modules (their src import closure) and remove unused plotting imports/methods; formal-v2 only observes model solves (and sets OutputFlag 0) and does not modify solver parameters or algorithm behavior. The result passes through src.utils.api._jsonable before reference_output, as api.solve does.", "solver_sha256": digest(attempt / "solver.py")})
    started = datetime.now(timezone.utc).isoformat()
    ok, stdout, stderr, output = executor(attempt, 60, 1)
    (attempt / "exec.log").write_text(stdout + "\n" + stderr)
    receipt_record = json.loads((attempt / "native_solver_receipt.json").read_text())
    receipt, receipt_request = receipt_record["result"], receipt_record["request"]
    verified = OutputVerifier.verify_output(suite, problem, source / "data.json", attempt / "output.json", 1e-6) if ok else {"ok": False, "issues": ["execution failed or exceeded 60-second wall limit"]}
    gold = OutputVerifier.load_gold_objective(source, suite)
    gold_monolithic = json.loads((source / "gold_monolithic.json").read_text()).get("best_objective") if (source / "gold_monolithic.json").exists() else None  # diagnostic only
    objective, _, _ = OutputVerifier.extract_objective(output)
    if verified.get("objective_value") is not None:
        objective = float(verified["objective_value"])
    status, gap = OutputVerifier.assess_status(exec_ok=ok, output=output, verify_ok=bool(verified["ok"]), objective=objective, gold_obj=gold, gap_threshold=GAP_THRESHOLD,
                                               execution_timed_out="[framework] solver execution timed out" in stderr)
    result = {"suite": suite, "problem": problem, "instance": instance, "variant": variant, "started_at_utc": started, "finished_at_utc": datetime.now(timezone.utc).isoformat(), "exec_ok": ok, "feasible_and_consistent": verified["ok"], "issues": verified.get("issues", []), "objective": objective, "gold": gold, "gold_monolithic": gold_monolithic,
              "gold_sources_agree": gold is not None and gold_monolithic is not None and abs(gold - gold_monolithic) <= 1e-9 * max(1.0, abs(gold)), "gap": gap, "numeric_status": status, "algorithm_review": "not performed (trusted formal reference); numeric_status is assess_status with verify_ok = numerical verification only", "passed_numeric": status == "ok", "execution_seconds": receipt.get("execution_seconds"), "timed_out": receipt.get("timed_out"), "returncode": receipt.get("returncode"),
              "native_cleanup_verified": receipt["cleanup_verified"], "integrity_violations": receipt["integrity_violations"], "input_sha256": digest(source / "data.json"), "scoring_inputs_sha256": scoring_inputs(source), "native_environment_sha256": hashlib.sha256(json.dumps(receipt_request["environment"], sort_keys=True).encode()).hexdigest(), "solver_sha256": digest(attempt / "solver.py"), "output_sha256": receipt.get("output_sha256"),
              "calibration": {key: value for key, value in ((output or {}).get("_calibration") or {}).items() if key != "model_solves"} | {"model_solve_count": len(((output or {}).get("_calibration") or {}).get("model_solves") or [])}, "solver_error_tail": stderr[-1200:] if not ok else None}
    save(result_path, result)
    print(json.dumps({key: result[key] for key in ("problem", "instance", "numeric_status", "execution_seconds", "gap", "issues")}), flush=True)
    if (result.get("execution_seconds") or 0) >= 5:
        time.sleep(10)
    return result


COLUMNS = ("suite", "problem", "instance", "numeric_status", "passed_numeric", "exec_ok", "feasible_and_consistent", "timed_out", "execution_seconds", "objective", "gold", "gap", "gold_sources_agree", "issues")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", default="formal-v2", choices=["formal-v2"])
    parser.add_argument("--problems", nargs="+", default=list(REFERENCES), choices=list(REFERENCES))
    parser.add_argument("--instances", nargs="*")
    args = parser.parse_args()
    destination = args.output.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    executor = NativeSandboxExecutor(license_file=license_file())
    selected = instances(args.problems, args.instances)
    results = [run_case(executor, destination, problem, instance, args.variant) for problem, instance in selected]
    tag = args.variant + ("_" + "_".join(args.problems) if args.problems != list(REFERENCES) else "") + ("_selected" if args.instances else "")
    save(destination / (tag + "_run_summary.json"), {"completed_at_utc": datetime.now(timezone.utc).isoformat(), "dataset": str(DATASET.relative_to(REPO)), "variant": args.variant, "timeout_seconds": 60, "threads": 1, "gap_threshold": GAP_THRESHOLD,
                                                    "native_environment": executor.environment, "native_environment_sha256": hashlib.sha256(json.dumps(executor.environment, sort_keys=True).encode()).hexdigest(),
                                                    "all_cases_same_environment": len({row["native_environment_sha256"] for row in results}) <= 1, "expected_cases": len(selected), "completed_cases": len(results), "cases": results})
    with (destination / (tag + "_run_summary.csv")).open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(COLUMNS)
        for row in results:
            writer.writerow([" | ".join(row[key]) if key == "issues" else row[key] for key in COLUMNS])
    by_family = defaultdict(list)
    for row in results:
        by_family[row["problem"]].append(row)
    for problem, rows in by_family.items():
        print(json.dumps({"problem": problem, "passed": sum(row["passed_numeric"] for row in rows), "cases": len(rows), "max_execution_seconds": max(row["execution_seconds"] or 0 for row in rows)}), flush=True)


if __name__ == "__main__":
    main()
