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


class HostCodexClient:
    """Fallback when the paper image's Codex CLI is rejected by the service (401 on workspace routing).

    Runs the host `codex exec` in an empty work directory with a throwaway CODEX_HOME holding only the
    login file and a minimal config: no web search, no user instructions, memories or skills. The prompt
    text is the same workspace-rules + task + context composition the Docker client sends.
    """

    def __init__(self, model, *, effort, compose):
        self.model, self.effort, self.compose = model, effort, compose

    def edit_file(self, target, prompt, *, context=None, log_dir=None, step="agent"):
        import os
        import subprocess
        import tempfile
        from types import SimpleNamespace

        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="dg-pilot-") as tmp:
            home, work = Path(tmp) / "home", Path(tmp) / "work"
            home.mkdir()
            work.mkdir()
            shutil.copyfile(Path.home() / ".codex/auth.json", home / "auth.json")
            (home / "auth.json").chmod(0o600)
            (home / "config.toml").write_text('web_search = "disabled"\n')
            (work / target.name).write_text(target.read_text() if target.exists() else "")
            text = self.compose(target.name, prompt, context)
            (log_dir / f"{step}_prompt.md").write_text(text)
            command = ["codex", "exec", "-m", self.model, "-c", f"model_reasoning_effort={self.effort}", "--sandbox", "workspace-write",
                       "--skip-git-repo-check", "--ephemeral", "--json", "-C", str(work), "-"]
            env = {k: v for k, v in os.environ.items() if not k.startswith(("CLAUDE", "DECOMPBENCH"))} | {"CODEX_HOME": str(home)}
            try:
                proc = subprocess.run(command, input=text, capture_output=True, text=True, timeout=900, env=env)
                rc, out, err = proc.returncode, proc.stdout, proc.stderr
            except subprocess.TimeoutExpired as exc:
                rc, out, err = -9, exc.stdout or "", "timeout"
            (log_dir / f"{step}_events.jsonl").write_text(out if isinstance(out, str) else out.decode(errors="replace"))
            (log_dir / f"{step}_stderr.log").write_text(err if isinstance(err, str) else err.decode(errors="replace"))
            produced = (work / target.name).read_text()
            target.write_text(produced)
        ok = rc == 0 and bool(produced.strip())
        return SimpleNamespace(ok=ok, error=None if ok else f"codex exec returncode {rc}")
RUNTIME_RECORD = Path(__file__).resolve().parents[3] / "DecompBench/outputs/runtime/runtime.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=sorted(SUITES), required=True)
    parser.add_argument("--instances", required=True, help="folder with <family>_<size>_seedNNN/data.json")
    parser.add_argument("--models", required=True, help="comma-separated Codex models")
    parser.add_argument("--out", required=True)
    parser.add_argument("--judge-model", default="gpt-6-astra")
    parser.add_argument("--host-codex", action="store_true", help="use the host Codex CLI with a clean CODEX_HOME instead of the paper image")
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

    def make(model, role, effort):
        client = docker(model, role, effort)
        return HostCodexClient(model, effort=effort, compose=client.compose_prompt) if args.host_codex else client

    judge = make(args.judge_model, "algorithm_judge", "high")
    screen = {r["instance"]: r for r in json.loads((Path(args.instances) / "screen.json").read_text())}
    rows = []
    for model in args.models.split(","):
        client = make(model, "generation", "medium")
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
            row = {"transport": "host-codex" if args.host_codex else "docker", "model": model, "instance": inst_dir.name, "generation_ok": bool(gen.ok), "generation_seconds": round(time.perf_counter() - started, 1)}
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
