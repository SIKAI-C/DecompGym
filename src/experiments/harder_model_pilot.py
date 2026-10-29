"""One-shot model pilot on a harder-family cohort: generation, sandboxed execution, scoring, judge.

Scoring follows the paper's runner: the same generation prompt builder, the shared OutputVerifier
(output contract + family checker + objective match), the shared status and gap rules, and the method
judge. Agent calls run only through the paper's Docker client. One attempt per (model, instance), no
retries; everything runs sequentially. Instances come from `01_gen.py --family-set harder` and need a
meta.json with a certified ground truth.
Run as: python -m src.experiments.harder_model_pilot --family gcol --instances data_harder/cg/gcol \\
            --models gpt-6.1-sol --out outputs/harder/pilot
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

from src.agent.docker_client import DockerCliClient
from src.agent.prompts.builder import PromptBuilder
from src.agent.prompts.judge import algorithm_review_prompt, parse_verdict
from src.experiments.docker_setup import runtime
from src.experiments.sandbox import NativeSandboxExecutor
from src.experiments.verification import OutputVerifier
from src.utils.errors import InfrastructureError


SUITES = {"lbbd": "bd", "vrptw": "cg", "pdptw": "cg", "gcol": "cg", "bppc": "cg", "nrp": "cg"}
REPOSITORY = Path(__file__).resolve().parents[2]
RUNTIME_RECORD = REPOSITORY / "outputs/runtime/runtime.json"
GAP_THRESHOLD, TOL = 0.01, 1e-6


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=sorted(SUITES), required=True)
    parser.add_argument("--instances", required=True, help="folder with <family>_<size>_seedNNN/{data.json,prompt.txt,meta.json}")
    parser.add_argument("--models", required=True, help="comma-separated Codex models")
    parser.add_argument("--out", required=True)
    parser.add_argument("--judge-model", default="gpt-6-astra")
    args = parser.parse_args()

    suite = SUITES[args.family]
    environment = json.loads(RUNTIME_RECORD.read_text())
    engine = runtime()
    engine.bind_environment(environment)
    executor = NativeSandboxExecutor(license_file=Path(environment["license_file"]), expected_environment=environment["native_environment"])
    out_root = Path(args.out)

    def docker(model, role, effort):
        return DockerCliClient(model, backend="codex-cli", image_id=environment["image_id"], runtime=engine,
                               evidence_root=out_root / "diagnostics", role=role, effort=effort, timeout=900)

    judge = docker(args.judge_model, "algorithm_judge", "high")
    scoring_code = {name: sha(REPOSITORY / name) for name in (f"src/{suite}/{args.family}/d_verifier.py", "src/utils/output_contract.py", "src/experiments/verification.py")}
    rows = []
    for model in args.models.split(","):
        client = docker(model, "generation", "medium")
        for inst_dir in sorted(p for p in Path(args.instances).iterdir() if (p / "data.json").exists()):
            meta_path = inst_dir / "meta.json"
            if not meta_path.exists():
                print(json.dumps({"model": model, "instance": inst_dir.name, "skipped": "no certified ground truth"}), flush=True)
                continue
            gold = OutputVerifier.load_gold_objective(inst_dir, suite)
            case = out_root / model / inst_dir.name
            attempt = case / "attempts" / "attempt_01"
            fingerprint = {"data": sha(inst_dir / "data.json"), "prompt": sha(inst_dir / "prompt.txt"), "meta": sha(meta_path),
                           "scoring_code": scoring_code, "model": model, "judge": args.judge_model}
            summary_path = attempt / "summary.json"
            if summary_path.exists():
                saved = json.loads(summary_path.read_text())
                if saved.get("fingerprint") == fingerprint:
                    rows.append(saved)
                    continue
                shutil.rmtree(case)  # scoring inputs changed: the old result is void
            attempt.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(inst_dir / "data.json", case / "data.json")
            prompt = PromptBuilder.build_generation_prompt((inst_dir / "prompt.txt").read_text(), solver_timeout=60, solver_threads=1)
            started = time.perf_counter()
            gen = client.edit_file(attempt / "solver.py", prompt, context=[case / "data.json"], log_dir=attempt / "calls_agent", step="agent")
            row = {"fingerprint": fingerprint, "transport": "docker", "model": model, "instance": inst_dir.name, "gold": gold,
                   "generation_ok": bool(gen.ok), "generation_seconds": round(time.perf_counter() - started, 1)}
            if not gen.ok and gen.terminal_state != "completed":
                raise InfrastructureError(f"{model} generation failed: {gen.error}")
            if gen.ok and (attempt / "solver.py").read_text().strip():
                exec_ok, _stdout, stderr, output = executor(attempt, 60, 1)
                receipt = json.loads((attempt / "native_solver_receipt.json").read_text())["result"]
                timed_out = "[framework] solver execution timed out" in stderr
                objective, issues, _ = OutputVerifier.extract_objective(output if isinstance(output, dict) else None)
                verification = {"ok": False, "issues": []}
                if exec_ok:
                    verification = OutputVerifier.verify_output(suite, args.family, case / "data.json", attempt / "output.json", TOL)
                    if verification.get("objective_value") is not None:
                        objective = float(verification["objective_value"])
                judged = judge.edit_file(attempt / "alg_verifier.md", algorithm_review_prompt(suite), context=[attempt / "solver.py"], log_dir=attempt / "calls_judge", step="llm_verifier")
                notes = (attempt / "alg_verifier.md").read_text().strip() if judged.ok else ""
                verdict = parse_verdict(notes) if notes else None
                if verdict is None:
                    raise InfrastructureError("Algorithm judge produced no parseable verdict")
                verify_ok = bool(verification["ok"]) and verdict == "pass"
                status, gap = OutputVerifier.assess_status(exec_ok=exec_ok, output=output if isinstance(output, dict) else None, verify_ok=verify_ok,
                                                           objective=objective, gold_obj=gold, gap_threshold=GAP_THRESHOLD, execution_timed_out=timed_out)
                row.update({"exec_ok": bool(exec_ok), "execution_seconds": receipt.get("execution_seconds"), "timed_out": timed_out,
                            "feasible": bool(verification["ok"]), "issues": (issues + verification.get("issues", []))[:3],
                            "reported_status": output.get("status") if isinstance(output, dict) else None, "objective": objective,
                            "gap_to_gold": gap, "judge_verdict": verdict, "status": status, "quality_pass": status == "ok"})
            summary_path.write_text(json.dumps(row, indent=1))
            rows.append(row)
            print(json.dumps({k: v for k, v in row.items() if k != "fingerprint"}), flush=True)
    (out_root / f"{args.family}_pilot.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
