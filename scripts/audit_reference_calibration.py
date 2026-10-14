"""Audit formal calibration receipts, certificates, and the preserved predecessor cohort."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path

from reference_calibration import CAMPAIGN, REPO, REFERENCES, OutputVerifier, digest, save, solver_source


def audit(root: Path):
    manifest = json.loads((CAMPAIGN / "campaign.json").read_text())
    spec = importlib.util.spec_from_file_location("prospective_output_contract", REPO / "src/agent/output_contract.py")
    contract = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(contract)
    results = []
    for case in manifest["instances"]:
        suite, problem, instance = (case[key] for key in ("suite", "problem", "instance"))
        path = root / "formal-v2" / suite / problem / instance
        result = json.loads((path / "calibration.json").read_text())
        attempt = path / "attempts/attempt_01"
        solver, inventory = solver_source(problem, "formal-v2")
        assert json.loads((path / "source_inventory.json").read_text())["original_sources"] == inventory
        expected_code = hashlib.sha256(solver.encode()).hexdigest()
        assert result["solver_sha256"] == digest(attempt / "solver.py") == expected_code
        original = CAMPAIGN / "dataset" / suite / problem / instance / "data.json"
        assert digest(original) == digest(REPO / "data" / suite / problem / instance / "data.json") == digest(path / "data.json") == result["input_sha256"]
        assert result["output_sha256"] == digest(attempt / "output.json")
        native = json.loads((attempt / "native_solver_receipt.json").read_text())
        assert native["request"]["environment"] == manifest["execution_environment"]["native_environment"]
        assert native["request"]["timeout"] == 60 and native["request"]["threads"] == 1
        receipt = native["result"]
        assert receipt["cleanup_verified"] and not receipt["integrity_violations"]
        assert receipt["exec_ok"] and receipt["returncode"] == 0 and not receipt["network_enabled"]
        assert receipt["execution_seconds"] < 60
        assert receipt["solver_sha256"] == expected_code
        output = json.loads((attempt / "output.json").read_text())
        data = json.loads(original.read_text())
        assert not contract.validate_output_contract(suite, problem, data, output), instance
        verified = OutputVerifier.verify_output(suite, problem, original, attempt / "output.json", 1e-6)
        assert verified["ok"], (instance, verified)
        objective = verified["objective_value"]
        gap = abs(objective - result["gold"]) / max(1, abs(result["gold"]))
        assert gap <= 0.01 and result["passed_numeric"], (instance, gap)
        evidence = output["_calibration"]
        formal = evidence["reference_result"]
        assert formal["reference_policy"] == "strict-calibrated-references-20261013-v2"
        assert formal["threads"] == 1 and formal["time_limit"] == 60
        assert len(evidence["model_solves"]) == len(formal["model_solves"]) > 0
        assert all(model["threads"] == 1 for model in evidence["model_solves"])
        assert all(model["threads"] == 1 and 0 <= model["time_limit"] <= 54 for model in formal["model_solves"])
        lower = evidence["global_lb"]
        if lower is not None:
            assert lower <= result["gold"] + 1e-6 * max(1, abs(result["gold"])), (instance, lower, result["gold"])
        if formal["proven_optimal"]:
            assert formal["status"] == "optimal" and formal["relative_gap"] <= formal["optimality_tolerance"]
            assert gap <= 1e-6, (instance, gap)
        else:
            assert formal["status"] == "feasible"
        if problem == "gap":
            assert formal["phase_one_records"][-1]["artificial_objective"] <= 1e-7
            assert all("feasible_seed" not in model["name"] for model in evidence["model_solves"])
        results.append(result | {"audited_status": formal["status"], "termination_reason": formal["termination_reason"]})
    assert len(results) == 81 and len({result["instance"] for result in results}) == 81
    families = {}
    for problem in REFERENCES:
        selected = [case for case in results if case["problem"] == problem]
        assert len(selected) == 9
        families[problem] = {"passed": 9, "total": 9, "max_execution_seconds": max(case["execution_seconds"] for case in selected), "max_gap": max(case["gap"] for case in selected), "statuses": dict(Counter(case["audited_status"] for case in selected))}
    summary = {"reference_policy": "strict-calibrated-references-20261013-v2", "comparison_protocol_id": manifest["protocol_id"], "passed": len(results), "total": 81, "max_execution_seconds": max(case["execution_seconds"] for case in results), "max_gap": max(case["gap"] for case in results), "families": families, "prospective_output_contract_sha256": digest(REPO / "src/agent/output_contract.py"), "checks": ["81 current formal source bundles and input/output hashes", "frozen native execution receipts and cleanup", "all observed models use one thread and share the formal time budget", "frozen numerical verifier and unchanged 1% threshold", "prospective output schema", "reported bounds checked against gold outside solver", "no false optimal labels in calibrated results"], "cases": results}
    save(root / "audited-summary.json", summary)
    return summary


def audit_preservation(path: Path):
    baseline = json.loads(path.read_text())
    workspace = REPO.parents[1]
    changed = [name for name, expected in baseline.items() if not (workspace / name).is_file() or digest(workspace / name) != expected]
    assert not changed, changed
    assert json.loads((CAMPAIGN / "pause_request.json").read_text())["resume_requires_explicit_user_request"]
    result = {"preserved_files": len(baseline), "changed": changed, "explicit_resumption_required": True, "predecessor_protocol_id": json.loads((CAMPAIGN / "campaign.json").read_text())["protocol_id"]}
    save(path.with_name("preservation-audit.json"), result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--preservation-inventory", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.root.resolve())
    print(json.dumps({key: result[key] for key in ("passed", "total", "max_execution_seconds", "max_gap", "families")}, indent=2))
    print(json.dumps(audit_preservation(args.preservation_inventory.resolve()), indent=2))


if __name__ == "__main__":
    main()
