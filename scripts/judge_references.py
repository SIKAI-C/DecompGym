"""Review each distinct calibrated reference with the prospective production judge."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time


REPO = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller", type=Path, default=REPO, help="Source checkout to inspect; defaults to this repository")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    controller, root = args.controller.resolve(), args.output.resolve()
    sys.path.insert(0, str(controller))
    from src.agent.algorithm_verifier import algorithm_review_prompt
    from src.agent.docker_client import DockerCliClient
    from src.execution_environment import load_execution_environment, require_backend_readiness
    from src.hybrid_runtime import runtime

    plan_path = REPO / "docs/2026-10-13-strict-decomposition-experiment-plan.json"
    plan = json.loads(plan_path.read_text())
    environment = load_execution_environment(plan, plan_path)
    readiness = require_backend_readiness(environment, ["codex-cli"])
    engine = runtime()
    engine.bind_environment(environment)
    if engine.image_id(environment["image_id"]) != environment["image_id"]:
        raise RuntimeError("The prepared judge image is unavailable")
    client = DockerCliClient(plan["execution"]["judge_model"], backend="codex-cli", image_id=environment["image_id"], runtime=engine, evidence_root=root / "runtime", effort=plan["execution"]["judge_effort"], timeout=plan["execution"]["agent_timeout"], role="algorithm_judge")
    baseline = REPO / "outputs/diagnostics/strict_reference_adoption_20261013/formal_final_v2"
    calibrated = json.loads((baseline / "audited-summary.json").read_text())
    assert calibrated["passed"] == calibrated["total"] == 81
    root.mkdir(parents=True, exist_ok=True)
    specification = {"client": client.protocol(), "judge_prompt_sha256": {suite: hashlib.sha256(algorithm_review_prompt(suite).encode()).hexdigest() for suite in ("bd", "cg")}, "calibration_summary_sha256": digest(baseline / "audited-summary.json"), "readiness": readiness, "scope": "One source-only judge call per distinct reference implementation; no solver execution or campaign resumption."}
    if (root / "manifest.json").exists():
        if json.loads((root / "manifest.json").read_text()) != specification:
            raise RuntimeError("Judge configuration changed; preserve this run and choose a new output directory")
    else:
        save(root / "manifest.json", specification)
    results = []
    for suite in ("bd", "cg"):
        (root / f"judge_{suite}.txt").write_text(algorithm_review_prompt(suite))
        for family in sorted((baseline / "formal-v2" / suite).iterdir()):
            sources = sorted(family.glob("*/attempts/attempt_01/solver.py"))
            hashes = {digest(source) for source in sources}
            assert len(sources) == 9 and len(hashes) == 1, family
            source = next(path for path in sources if "medium_seed000" in path.parts[-4])
            inventory_path = source.parents[2] / "source_inventory.json"
            inventory = json.loads(inventory_path.read_text())["original_sources"]
            assert all(digest(REPO / name) == expected == digest(controller / name) for name, expected in inventory.items())
            directory = root / suite / family.name
            directory.mkdir(parents=True, exist_ok=True)
            solver = directory / "solver.py"
            if solver.exists():
                assert digest(solver) in hashes
            else:
                shutil.copyfile(source, solver)
            target = directory / "alg_verifier.md"
            target.touch(exist_ok=True)
            request = {"suite": suite, "problem": family.name, "source": str(source), "source_sha256": digest(solver), "source_inventory": inventory, "covered_instances": [path.parts[-4] for path in sources], "judge_prompt_sha256": specification["judge_prompt_sha256"][suite]}
            if (directory / "request.json").exists():
                assert json.loads((directory / "request.json").read_text()) == request
            else:
                save(directory / "request.json", request)
            if (directory / "result.json").exists():
                result = json.loads((directory / "result.json").read_text())
                assert digest(target) == result["verdict_sha256"]
                results.append(result)
                continue
            started = time.perf_counter()
            print(json.dumps({"state": "judging", "suite": suite, "problem": family.name, "model": client.model, "effort": client.effort}), flush=True)
            if (directory / "llm_verifier_container.json").exists():
                answer = client.recover_file(target, log_dir=directory, step="llm_verifier")
            else:
                answer = client.edit_file(target, algorithm_review_prompt(suite), context=[solver], log_dir=directory, step="llm_verifier")
            notes = target.read_text().strip()
            first = notes.splitlines()[0].strip().upper() if notes else ""
            verdict = {"VERDICT: PASS": "pass", "VERDICT: FAIL": "fail"}.get(first)
            assert digest(solver) == request["source_sha256"]
            result = {**request, "finished_at_utc": datetime.now(timezone.utc).isoformat(), "verdict": verdict, "verdict_sha256": digest(target), "notes": notes, "metrics": asdict(answer), "total_seconds": time.perf_counter() - started}
            save(directory / "result.json", result)
            results.append(result)
            save(root / "summary.json", {"complete": len(results) == 9, "passed": sum(item["verdict"] == "pass" for item in results), "total": len(results), "results": results})
            print(json.dumps({"state": "reviewed", "problem": family.name, "verdict": verdict, "ok": answer.ok, "seconds": answer.elapsed_seconds, "usage": answer.usage}), flush=True)
            if not answer.ok or verdict is None:
                raise RuntimeError("Judge infrastructure or verdict delivery failed; preserve evidence before deciding how to recover")
    save(root / "summary.json", {"complete": True, "passed": sum(item["verdict"] == "pass" for item in results), "total": 9, "results": results})


if __name__ == "__main__":
    main()
