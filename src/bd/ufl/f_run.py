from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from aider.coders import Coder
from aider.io import InputOutput
from aider.models import Model

from utils.time_utils import get_current_time


class UFLRunner:
    """Run a single UFL instance: use aider to write solver.py, execute it, compare to ground truth."""

    DEFAULT_SOLVER_TIMEOUT = 300  # seconds

    def __init__(
        self,
        folder: Path,
        model_name: str,
        *,
        auto_yes: bool = True,
        stream: bool = False,
        tol: float = 1e-6,
        solver_timeout: int = DEFAULT_SOLVER_TIMEOUT,
    ):
        self.folder = folder
        self.model_name = model_name
        self.auto_yes = auto_yes
        self.stream = stream
        self.tol = tol
        self.solver_timeout = solver_timeout

        self.model = Model(model_name)
        self.io = InputOutput(yes=auto_yes)

    def _safe_load_json(self, path: Path) -> Optional[Dict[str, Any]]:
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text())
        except Exception as e:
            print(f"[ERROR] Failed to parse JSON from {path}: {e}")
            return None

    def _run_aider(self, prompt: str) -> None:
        solver_path = self.folder / "solver.py"
        if not solver_path.exists():
            solver_path.write_text("")

        fnames = [str(solver_path)]
        read_only_fnames = [
            str((self.folder / "data.json").resolve()),
            str((self.folder / "prompt.txt").resolve()),
        ]

        coder = Coder.create(
            main_model=self.model,
            fnames=fnames,
            read_only_fnames=read_only_fnames,
            io=self.io,
            stream=self.stream,
            use_git=False,
            edit_format="diff",
        )
        coder.run(prompt)

    def _run_solver(self) -> Dict[str, Any]:
        solver_path = self.folder / "solver.py"
        try:
            proc = subprocess.run(
                ["python", str(solver_path)],
                cwd=self.folder,
                capture_output=True,
                text=True,
                check=False,
                timeout=self.solver_timeout,
            )
        except subprocess.TimeoutExpired:
            return {
                "status": "time_limit_reached",
                "error": f"Solver exceeded time limit of {self.solver_timeout} seconds",
                "timeout_seconds": self.solver_timeout,
            }
        except Exception as e:
            return {
                "status": "solver_crash",
                "error": f"Exception when running solver.py: {e}",
            }

        if proc.returncode != 0:
            return {
                "status": "solver_crash",
                "returncode": proc.returncode,
                "stdout": proc.stdout[-2000:],
                "stderr": proc.stderr[-2000:],
            }

        output_path = self.folder / "output.json"
        if not output_path.exists():
            return {
                "status": "no_output",
                "error": "output.json not found after running solver.py",
                "stdout": proc.stdout[-2000:],
                "stderr": proc.stderr[-2000:],
            }

        output = self._safe_load_json(output_path)
        if output is None:
            return {
                "status": "invalid_output_json",
                "error": "Failed to parse output.json",
            }

        return {"status": "ok", "output": output}

    def _compare_with_ground_truth(self, output: Dict[str, Any]) -> Dict[str, Any]:
        obj = output.get("objective_value", None)

        gt_mono_path = self.folder / "gold_monolithic.json"
        gt_bd_path = self.folder / "gold_bd.json"

        gt_mono = self._safe_load_json(gt_mono_path)
        gt_bd = self._safe_load_json(gt_bd_path)

        res: Dict[str, Any] = {
            "objective": obj,
            "gt_monolithic": None,
            "gt_bd": None,
            "gap_monolithic": None,
            "gap_bd": None,
            "match_monolithic": None,
            "match_bd": None,
        }

        if obj is None:
            res["status"] = "ok_no_objective"
            return res

        if gt_mono is not None:
            z_star = gt_mono.get("best_objective") or gt_mono.get("objective_value")
            res["gt_monolithic"] = z_star
            if z_star is not None:
                gap = abs(obj - z_star)
                rel_gap = gap / max(1.0, abs(z_star))
                res["gap_monolithic"] = rel_gap
                res["match_monolithic"] = rel_gap <= self.tol

        if gt_bd is not None:
            z_bd = gt_bd.get("best_objective") or gt_bd.get("objective_value")
            res["gt_bd"] = z_bd
            if z_bd is not None:
                gap = abs(obj - z_bd)
                rel_gap = gap / max(1.0, abs(z_bd))
                res["gap_bd"] = rel_gap
                res["match_bd"] = rel_gap <= self.tol

        if res["match_monolithic"] is True:
            res["status"] = "ok_match_gt"
            res["matched_against"] = "monolithic"
        elif res["match_bd"] is True:
            res["status"] = "ok_match_gt"
            res["matched_against"] = "bd"
        elif gt_mono is None and gt_bd is None:
            res["status"] = "ok_no_gt"
        else:
            res["status"] = "ok_obj_mismatch"

        return res

    def run(self) -> Dict[str, Any]:
        prompt_path = self.folder / "prompt.txt"
        if not prompt_path.exists():
            raise FileNotFoundError(f"prompt.txt not found in {self.folder}")

        prompt = prompt_path.read_text()

        # 1) Ask aider to (re)write solver.py
        self._run_aider(prompt)

        # 2) Run solver.py and inspect output.json
        solver_result = self._run_solver()
        if solver_result["status"] != "ok":
            print(f"[UFL] Status: {solver_result['status']}")
            return solver_result

        # 3) Compare against ground truth if available
        compare_result = self._compare_with_ground_truth(solver_result["output"])

        # Pretty-print short summary
        print(f"[UFL] status={compare_result['status']}")
        print(f"[UFL] objective={compare_result['objective']}")
        if compare_result.get("gt_monolithic") is not None:
            print(
                f"[UFL] gt_monolithic={compare_result['gt_monolithic']}, "
                f"gap={compare_result['gap_monolithic']}"
            )
        if compare_result.get("gt_bd") is not None:
            print(
                f"[UFL] gt_bd={compare_result['gt_bd']}, "
                f"gap={compare_result['gap_bd']}"
            )

        return compare_result


def find_instance_dirs(root: Path) -> List[Path]:
    """Find all subdirectories of root that look like UFL instances (data.json + prompt.txt)."""
    if not root.exists():
        raise FileNotFoundError(f"Instances root {root} does not exist")

    dirs: List[Path] = []
    for p in sorted(root.iterdir()):
        if p.is_dir() and (p / "data.json").exists() and (p / "prompt.txt").exists():
            dirs.append(p)
    return dirs


def create_run_folder(instances_root: Path, runs_root: Path, model_name: str) -> Path:
    """
    Create a timestamped run folder under runs_root and copy all instances into it.
    Ensures each copied instance has a solver.py placeholder.
    """
    runs_root.mkdir(parents=True, exist_ok=True)
    timestamp = get_current_time()
    model_safe = model_name.replace("/", "_").replace(":", "_")
    run_folder_name = f"ufl_{timestamp}_{model_safe}"
    run_folder = runs_root / run_folder_name

    print(f"[UFL] Creating run folder: {run_folder}")
    run_folder.mkdir(parents=True, exist_ok=False)

    inst_dirs = find_instance_dirs(instances_root)
    print(f"[UFL] Copying {len(inst_dirs)} instances to run folder...")
    for inst_dir in inst_dirs:
        dest = run_folder / inst_dir.name
        shutil.copytree(inst_dir, dest)
        solver_path = dest / "solver.py"
        if not solver_path.exists():
            solver_path.write_text("")  # placeholder for aider

    index_path = instances_root / "index.json"
    if index_path.exists():
        shutil.copy(index_path, run_folder / "index.json")

    run_meta = {
        "timestamp": timestamp,
        "model_name": model_name,
        "source_instances_root": str(instances_root),
        "num_instances": len(inst_dirs),
    }
    (run_folder / "run_meta.json").write_text(json.dumps(run_meta, indent=2))

    return run_folder


def create_single_run_folder(instance_folder: Path, runs_root: Path, model_name: str) -> Path:
    """
    Create a timestamped run folder and copy a single instance into it.
    Ensures solver.py exists (placeholder if missing).
    """
    runs_root.mkdir(parents=True, exist_ok=True)
    timestamp = get_current_time()
    model_safe = model_name.replace("/", "_").replace(":", "_")
    run_folder_name = f"ufl_{timestamp}_{model_safe}"
    run_folder = runs_root / run_folder_name

    print(f"[UFL] Creating single-instance run folder: {run_folder}")
    run_folder.mkdir(parents=True, exist_ok=False)

    dest = run_folder / instance_folder.name
    shutil.copytree(instance_folder, dest)
    solver_path = dest / "solver.py"
    if not solver_path.exists():
        solver_path.write_text("")  # placeholder for aider

    run_meta = {
        "timestamp": timestamp,
        "model_name": model_name,
        "source_instance": str(instance_folder),
        "num_instances": 1,
    }
    (run_folder / "run_meta.json").write_text(json.dumps(run_meta, indent=2))

    return dest


def run_single_instance(
    folder: Path,
    model_name: str,
    *,
    auto_yes: bool = True,
    stream: bool = False,
    tol: float = 1e-6,
    solver_timeout: int = UFLRunner.DEFAULT_SOLVER_TIMEOUT,
) -> Dict[str, Any]:
    runner = UFLRunner(folder, model_name, auto_yes=auto_yes, stream=stream, tol=tol, solver_timeout=solver_timeout)
    print(f"[UFL] Running UFLRunner in {folder}")
    return runner.run()


def run_benchmark_suite(
    instances_root: Path,
    model_name: str,
    *,
    runs_root: Optional[Path] = None,
    auto_yes: bool = True,
    stream: bool = False,
    tol: float = 1e-6,
    solver_timeout: int = UFLRunner.DEFAULT_SOLVER_TIMEOUT,
) -> Dict[str, Any]:
    inst_dirs = find_instance_dirs(instances_root)
    if not inst_dirs:
        raise RuntimeError(f"No instance directories with data.json + prompt.txt found under {instances_root}")

    print(f"[UFL] Found {len(inst_dirs)} instances under {instances_root}")
    print(f"[UFL] Solver timeout: {solver_timeout} seconds")

    if runs_root is None:
        runs_root = Path("z_artifacts/output/ufl")
    run_folder = create_run_folder(instances_root, runs_root, model_name)
    working_root = run_folder
    print(f"[UFL] Running benchmark in: {working_root}")

    working_inst_dirs = find_instance_dirs(working_root)

    results: List[Dict[str, Any]] = []
    num_match = 0
    for d in working_inst_dirs:
        print(f"\n[UFL] === Instance: {d.name} ===")
        runner = UFLRunner(d, model_name, auto_yes=auto_yes, stream=stream, tol=tol, solver_timeout=solver_timeout)
        res = runner.run()
        res_with_meta: Dict[str, Any] = {"instance_dir": str(d), "instance_name": d.name, "result": res}
        results.append(res_with_meta)

        status = res.get("status")
        if status == "ok_match_gt":
            num_match += 1

    summary = {
        "run_folder": str(run_folder),
        "source_instances_root": str(instances_root),
        "model_name": model_name,
        "solver_timeout_seconds": solver_timeout,
        "timestamp": Path(run_folder).name,
        "num_instances": len(results),
        "num_match_ground_truth": num_match,
        "match_rate": num_match / len(results) if results else 0.0,
        "results": results,
    }

    summary_path = working_root / "benchmark_results.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"\n{'='*60}")
    print("[UFL] BENCHMARK COMPLETE")
    print(f"{'='*60}")
    print(f"[UFL] Matched ground truth: {num_match}/{len(results)} ({100*num_match/len(results):.1f}%)")
    print(f"[UFL] Results written to: {summary_path}")
    print(f"[UFL] Run folder: {working_root}")

    # Note: E/V/Q analysis intentionally omitted for now.

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run LLM-based Benders benchmark for UFL (Uncapacitated Facility Location).\n\n"
            "Typical workflows:\n"
            "  # Single instance in current folder\n"
            "  python -m ufl.f_run --folder ./z_artifacts/benchmarks/ufl/ufl_easy_seed000 --model azure/your-model\n\n"
            "  # Suite of instances (with trace preservation under z_artifacts/output)\n"
            "  python -m ufl.f_run --instances-root ./z_artifacts/benchmarks/ufl --model azure/your-model\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="azure/adeza3-DE00023575-gpt-4.1",
        help="Model identifier passed to aider.",
    )
    parser.add_argument(
        "--folder",
        type=str,
        default=None,
        help="Directory of a single instance (containing data.json and prompt.txt). If set, runs only that instance.",
    )
    parser.add_argument(
        "--instances-root",
        type=str,
        default="z_artifacts/benchmarks/ufl",
        help="Root directory containing multiple instance folders. If provided, runs a suite.",
    )
    parser.add_argument(
        "--runs-root",
        type=str,
        default="z_artifacts/output/ufl",
        help="Root directory to store run traces. Default: z_artifacts/output/ufl (timestamped subfolder created).",
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable streaming output from aider.",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-6,
        help="Relative tolerance for matching ground truth objectives.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=UFLRunner.DEFAULT_SOLVER_TIMEOUT,
        help=f"Timeout in seconds for running solver.py (default: {UFLRunner.DEFAULT_SOLVER_TIMEOUT}s).",
    )

    args = parser.parse_args()

    model_name = args.model
    auto_yes = True
    stream = not args.no_stream

    runs_root = Path(args.runs_root).resolve() if args.runs_root else Path("z_artifacts/output/ufl")

    if args.folder:
        folder = Path(args.folder).resolve()
        # Copy the single instance into a timestamped run folder
        single_run_inst = create_single_run_folder(folder, runs_root, model_name)
        run_single_instance(
            folder=single_run_inst,
            model_name=model_name,
            auto_yes=auto_yes,
            stream=stream,
            tol=args.tol,
            solver_timeout=args.timeout,
        )
    else:
        instances_root = Path(args.instances_root).resolve()
        run_benchmark_suite(
            instances_root=instances_root,
            model_name=model_name,
            runs_root=runs_root,
            auto_yes=auto_yes,
            stream=stream,
            tol=args.tol,
            solver_timeout=args.timeout,
        )


if __name__ == "__main__":
    main()

