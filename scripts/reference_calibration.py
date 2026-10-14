"""Calibrate saved reference algorithms independently of the paused agent campaign."""
from __future__ import annotations

import argparse
import ast
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time


REPO = Path(__file__).resolve().parents[1]
CAMPAIGN = REPO / "outputs/experiments/fixed81_qualitative_20261012_r1"
sys.path.insert(0, str(CAMPAIGN / "controller"))
from src.agent.native_solver import NativeSandboxExecutor
from src.agent.output_verifier import OutputVerifier


REFERENCES = {
    "ufl": ("bd", "b_gold_bd", "UFLBenders"),
    "fctp": ("bd", "a_gold_bd", "FCTPBenders"),
    "km": ("bd", "a_gold_bd", "KMedianBenders"),
    "lrp": ("bd", "a_gold_bd", "LRPBenders"),
    "tsso": ("bd", "a_gold_bd", "TSSOBenders"),
    "bp": ("cg", "a_gold_cg", "BinPackingColumnGeneration"),
    "csp": ("cg", "a_gold_cg", "CuttingStockColumnGeneration"),
    "gap": ("cg", "a_gold_cg", "GAPColumnGeneration"),
    "shift": ("cg", "a_gold_cg", "ShiftColumnGeneration"),
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
        if node.name in {"plot_bounds", "plot_instance", "plot_solution", "_ensure_ax"}:
            return None
        return self.generic_visit(node)


def bundle_sources(problem):
    suite, module, cls = REFERENCES[problem]
    names = ["src.utils.base_data", "src.utils.reference_runtime", "src.utils.base_bd_class" if suite == "bd" else "src.utils.base_cg_class", f"src.{suite}.{problem}.a_gen_data", f"src.{suite}.{problem}.{module}", "src.utils.reference_output"]
    sources, inventory = {}, {}
    for name in names:
        path = REPO / (name.replace(".", "/") + ".py")
        original = path.read_text()
        tree = RemoveUnusedPlots().visit(ast.parse(original))
        ast.fix_missing_locations(tree)
        sources[name] = ast.unparse(tree) + "\n"
        inventory[str(path.relative_to(REPO))] = digest(path)
    return sources, inventory, f"src.{suite}.{problem}.{module}", cls


def solver_source(problem, variant):
    sources, inventory, module, cls = bundle_sources(problem)
    suite = REFERENCES[problem][0]
    variants = REPO / "scripts/reference_calibration_variants.py"
    variant_source = variants.read_text() if variant not in {"existing", "formal-v2"} and variants.exists() else ""
    source = f'''from __future__ import annotations
import time
STARTED = time.perf_counter()
import json, sys, types, math
from pathlib import Path
import gurobipy as gp
from gurobipy import GRB
SOURCES = {sources!r}
VARIANT = {variant!r}
PROBLEM = {problem!r}
SUITE = {suite!r}
DEADLINE = STARTED + 54.0
MODEL_SOLVES = []
class CalibrationDeadline(Exception):
    pass
OriginalModel = gp.Model
class ObservedModel(OriginalModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.Params.OutputFlag = 0
        if VARIANT != "formal-v2":
            self.Params.Threads = 1
    def optimize(self, *args, **kwargs):
        if VARIANT not in {{"existing", "formal-v2"}}:
            remaining = DEADLINE - time.monotonic()
            if remaining <= 0:
                raise CalibrationDeadline()
            self.Params.TimeLimit = min(self.Params.TimeLimit, remaining)
        if VARIANT != "formal-v2":
            self.Params.Threads = 1
        started = time.monotonic()
        super().optimize(*args, **kwargs)
        MODEL_SOLVES.append({{"name": self.ModelName, "threads": self.Params.Threads, "seconds": time.monotonic() - started, "status": self.Status, "variables": self.NumVars, "constraints": self.NumConstrs}})
        if VARIANT not in {{"existing", "formal-v2"}} and time.monotonic() >= DEADLINE:
            raise CalibrationDeadline()
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
def publish(result, termination='reference_return'):
    value = reference_output(SUITE, PROBLEM, DATA, result)
    value.setdefault('status', 'feasible')
    value.setdefault('method', 'benders_decomposition' if SUITE == 'bd' else 'column_generation')
    value['_calibration'] = {{'variant': VARIANT, 'termination': termination, 'whole_script_seconds': time.monotonic() - STARTED, 'reported_loop_seconds': result.get('total_time'), 'iterations': result.get('iters'), 'model_solves': MODEL_SOLVES, 'uses_numpy': True, 'global_lb': result.get('global_lb') if result.get('global_lb') is not None and math.isfinite(result['global_lb']) else None}}
    if VARIANT == 'formal-v2':
        value['_calibration']['reference_result'] = {{key: result.get(key) for key in ('reference_policy', 'termination_reason', 'relative_gap', 'target_gap', 'target_gap_met', 'proven_optimal', 'optimality_tolerance', 'status', 'threads', 'time_limit', 'model_solves', 'phase_one_records')}}
    Path('output.json').write_text(json.dumps(value, allow_nan=False, default=lambda value: value.item() if hasattr(value,'item') else value.tolist()))
{variant_source}
if VARIANT == 'formal-v2':
    instance = ReferenceClass(verbose=False, time_limit=60.0)
    publish(instance.run('../../data.json', started_at=STARTED), 'formal_reference_return')
elif VARIANT == 'existing':
    instance = ReferenceClass(verbose=False)
    publish(instance.run('../../data.json'))
else:
    run_variant()
'''
    if variant_source:
        inventory[str(variants.relative_to(REPO))] = digest(variants)
    return source, inventory


def historical_inventory(destination):
    records = []
    for problem, (suite, _, _) in REFERENCES.items():
        for path in sorted((REPO / "data" / suite / problem).glob("*/data.json")):
            instance = path.parent.name
            canonical = CAMPAIGN / "dataset" / suite / problem / instance
            assert digest(path) == digest(canonical / "data.json")
            result = json.loads((path.parent / ("gold_bd.json" if suite == "bd" else "gold_cg.json")).read_text())
            gold = json.loads((canonical / "gold_monolithic.json").read_text())["best_objective"]
            records.append({"suite": suite, "problem": problem, "instance": instance, "recorded_loop_seconds": result["total_time"], "objective": result["best_objective"], "gold": gold, "gap": abs(result["best_objective"] - gold) / max(1, abs(gold)), "input_sha256": digest(path), "compliant_initialization": problem != "gap"})
    save(destination / "historical_inventory.json", records)


def run_case(executor, destination, problem, instance, variant):
    suite = REFERENCES[problem][0]
    source = CAMPAIGN / "dataset" / suite / problem / instance
    case = destination / variant / suite / problem / instance
    result_path = case / "calibration.json"
    solver, inventory = solver_source(problem, variant)
    if result_path.exists():
        previous = json.loads(result_path.read_text())
        previous_inventory = json.loads((case / "source_inventory.json").read_text())
        expected_solver = hashlib.sha256(solver.encode()).hexdigest()
        if previous_inventory["original_sources"] != inventory or previous["input_sha256"] != digest(source / "data.json") or previous["solver_sha256"] != expected_solver or digest(case / "attempts/attempt_01/solver.py") != expected_solver:
            raise RuntimeError("Calibration inputs/source changed; use a new variant or output directory and preserve the old evidence")
        if previous.get("output_sha256") and digest(case / "attempts/attempt_01/output.json") != previous["output_sha256"]:
            raise RuntimeError("Saved calibration output changed")
        return previous
    attempt = case / "attempts/attempt_01"
    attempt.mkdir(parents=True)
    shutil.copyfile(source / "data.json", case / "data.json")
    (attempt / "solver.py").write_text(solver)
    save(case / "source_inventory.json", {"original_sources": inventory, "adapter": "Inline formal reference modules and remove unused plotting imports/methods; formal-v2 only observes model solves and does not modify solver parameters or algorithm behavior. Other modes are historical calibration adapters.", "solver_sha256": digest(attempt / "solver.py")})
    started = datetime.now(timezone.utc).isoformat()
    ok, stdout, stderr, output = executor(attempt, 60, 1)
    (attempt / "exec.log").write_text(stdout + "\n" + stderr)
    receipt = json.loads((attempt / "native_solver_receipt.json").read_text())["result"]
    verified = OutputVerifier.verify_output(suite, problem, source / "data.json", attempt / "output.json", 1e-6) if ok else {"ok": False, "issues": ["execution failed or exceeded 60-second wall limit"]}
    gold = json.loads((source / "gold_monolithic.json").read_text())["best_objective"]
    objective = verified.get("objective_value")
    if objective is None and output:
        objective = output.get("objective_value")
    gap = abs(objective - gold) / max(1, abs(gold)) if objective is not None else None
    result = {"suite": suite, "problem": problem, "instance": instance, "variant": variant, "started_at_utc": started, "finished_at_utc": datetime.now(timezone.utc).isoformat(), "exec_ok": ok, "feasible_and_consistent": verified["ok"], "issues": verified.get("issues", []), "objective": objective, "gold": gold, "gap": gap, "passed_numeric": bool(ok and verified["ok"] and gap is not None and gap <= .01), "method_compliant": not (problem == "gap" and variant == "existing"), "execution_seconds": receipt.get("execution_seconds"), "native_cleanup_verified": receipt["cleanup_verified"], "integrity_violations": receipt["integrity_violations"], "input_sha256": digest(source / "data.json"), "solver_sha256": digest(attempt / "solver.py"), "output_sha256": receipt.get("output_sha256"), "calibration": (output or {}).get("_calibration"), "solver_error_tail": stderr[-1200:] if not ok else None}
    save(result_path, result)
    print(json.dumps({key: result[key] for key in ("problem", "instance", "variant", "passed_numeric", "method_compliant", "execution_seconds", "gap", "issues")}), flush=True)
    if result.get("execution_seconds", 0) >= 5:
        time.sleep(10)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", default="existing")
    parser.add_argument("--problems", nargs="+", default=list(REFERENCES))
    parser.add_argument("--instances", nargs="*")
    args = parser.parse_args()
    destination = args.output.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((CAMPAIGN / "campaign.json").read_text())
    assert json.loads((CAMPAIGN / "pause_request.json").read_text())["resume_requires_explicit_user_request"]
    historical_inventory(destination)
    executor = NativeSandboxExecutor(license_file=Path(manifest["execution_environment"]["license_file"]), expected_environment=manifest["execution_environment"]["native_environment"])
    results = []
    for case in manifest["instances"]:
        if case["problem"] not in args.problems or args.instances and case["instance"] not in args.instances:
            continue
        if case["problem"] == "gap" and args.variant == "existing":
            continue
        results.append(run_case(executor, destination, case["problem"], case["instance"], args.variant))
    save(destination / (args.variant + "_run_summary.json"), {"completed_at_utc": datetime.now(timezone.utc).isoformat(), "campaign_protocol": manifest["protocol_id"], "variant": args.variant, "cases": results})


if __name__ == "__main__":
    main()
