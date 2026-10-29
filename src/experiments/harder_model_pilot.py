"""One-shot model pilot on a candidate harder family: generation, sandboxed execution, verification, judge.

Uses the paper's Docker agent client, native solver sandbox and method judge, with one attempt per
(model, instance) and no retries. Everything runs sequentially.
Run as: python -m src.experiments.harder_model_pilot --family lbbd --instances outputs/harder/lbbd \\
            --models gpt-6.1-sol --out outputs/harder/pilot
"""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import shutil
import time

from src.agent.docker_client import DockerCliClient
from src.agent.prompts.builder import PromptBuilder
from src.agent.prompts.judge import algorithm_review_prompt, parse_verdict
from src.experiments.docker_setup import runtime
from src.experiments.sandbox import NativeSandboxExecutor


SUITES = {"lbbd": "bd", "sslp": "bd", "vrptw": "cg"}
RUNTIME_RECORD = Path(__file__).resolve().parents[2] / "outputs/runtime/runtime.json"



def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=sorted(SUITES), required=True)
    parser.add_argument("--instances", required=True, help="folder with <family>_<size>_seedNNN/data.json")
    parser.add_argument("--models", required=True, help="comma-separated Codex models")
    parser.add_argument("--out", required=True)
    parser.add_argument("--judge-model", default="gpt-6-astra")
    args = parser.parse_args()

    suite = SUITES[args.family]
    package = f"src.{suite}.{args.family}"
    render_prompt = importlib.import_module(f"{package}.b_problem").render_prompt
    verify = importlib.import_module(f"{package}.d_verifier").verify_solution
    environment = json.loads(RUNTIME_RECORD.read_text())
    engine = runtime()
    engine.bind_environment(environment)
    executor = NativeSandboxExecutor(license_file=Path(environment["license_file"]), expected_environment=environment["native_environment"])
    out_root = Path(args.out)
    def docker(model, role, effort):
        return DockerCliClient(model, backend="codex-cli", image_id=environment["image_id"], runtime=engine,
                               evidence_root=out_root / "diagnostics", role=role, effort=effort, timeout=900)

    judge = docker(args.judge_model, "algorithm_judge", "high")
    screen = {r["instance"]: r for r in json.loads((Path(args.instances) / "screen.json").read_text())}
    rows = []
    for model in args.models.split(","):
        client = docker(model, "generation", "medium")
        for inst_dir in sorted(p for p in Path(args.instances).iterdir() if (p / "data.json").exists()):
            case = out_root / model / inst_dir.name
            attempt = case / "attempts" / "attempt_01"
            summary_path = attempt / "summary.json"
            if summary_path.exists():
                rows.append(json.loads(summary_path.read_text()))
                continue
            attempt.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(inst_dir / "data.json", case / "data.json")
            data = json.loads((case / "data.json").read_text())
            prompt = PromptBuilder.build_generation_prompt(render_prompt(data), solver_timeout=60, solver_threads=1)
            (case / "prompt.txt").write_text(prompt)
            started = time.perf_counter()
            gen = client.edit_file(attempt / "solver.py", prompt, context=[case / "data.json"], log_dir=attempt / "calls_agent", step="agent")
            row = {"transport": "docker", "model": model, "instance": inst_dir.name, "generation_ok": bool(gen.ok), "generation_seconds": round(time.perf_counter() - started, 1)}
            if gen.ok and (attempt / "solver.py").read_text().strip():
                exec_ok, _stdout, stderr, output = executor(attempt, 60, 1)
                row["exec_ok"] = bool(exec_ok)
                check = verify(case / "data.json", output) if exec_ok and isinstance(output, dict) else {"ok": False, "issues": [stderr[-300:]], "objective_value": None}
                opt = screen[inst_dir.name]["monolithic"].get("obj")
                obj = check["objective_value"]
                row.update({"feasible": check["ok"], "objective": obj, "optimum": opt, "reported_status": (output or {}).get("status") if isinstance(output, dict) else None,
                            "gap_to_opt": abs(obj - opt) / max(1.0, abs(opt)) if check["ok"] and opt is not None else None, "issues": check["issues"][:2]})
                judged = judge.edit_file(attempt / "alg_verifier.md", algorithm_review_prompt(suite), context=[attempt / "solver.py"], log_dir=attempt / "calls_judge", step="llm_verifier")
                notes = (attempt / "alg_verifier.md").read_text() if judged.ok else ""
                row["judge_verdict"] = parse_verdict(notes) if notes else None
                row["unproven_optimal_claim"] = row["reported_status"] == "optimal" and row["judge_verdict"] != "pass"
                row["quality_pass"] = bool(row["feasible"] and row["gap_to_opt"] is not None and row["gap_to_opt"] <= 0.01 and row["judge_verdict"] == "pass")
            summary_path.write_text(json.dumps(row, indent=1))
            rows.append(row)
            print(json.dumps(row), flush=True)
    (out_root / f"{args.family}_pilot.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
