"""Run isolated, immutable LRP calibration experiments without model/API calls.

Example: python -m src.pilots.lrp.run --config configs/lrp_pilot_smoke.json --output outputs/lrp-smoke
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[3]


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_config(config):
    from .solvers import DEFAULT_METHODS, METHODS

    if config.get("schema_version") != 1:
        raise ValueError("Expected schema_version 1")
    methods = config.get("methods", list(DEFAULT_METHODS))
    if not methods or len(set(methods)) != len(methods) or any(m not in METHODS for m in methods):
        raise ValueError("Invalid or duplicate methods")
    if not config.get("cases"):
        raise ValueError("No cases configured")
    names = [case["name"] for case in config["cases"]]
    if len(names) != len(set(names)) or any(
        not n or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for c in n) for n in names
    ):
        raise ValueError("Use unique lowercase case names containing letters, digits, hyphens or underscores")
    for case in config["cases"]:
        seeds = case.get("seeds", [0])
        if not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int or s < 0 for s in seeds):
            raise ValueError("Seeds must be distinct nonnegative integers")
        if "source" in case and seeds != [0]:
            raise ValueError("Imported inputs use seeds [0]; seeds do not regenerate existing data")
    seconds = config.get("solver_seconds", 15.0)
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not 0 < seconds <= 3600:
        raise ValueError("solver_seconds must be in (0, 3600]")
    iterations = config.get("max_iterations", 1000)
    if type(iterations) is not int or not 1 <= iterations <= 100000:
        raise ValueError("Invalid max_iterations")


def worker(request_path):
    import resource

    from .solvers import Solver

    request = json.loads(Path(request_path).read_text())
    folder = Path(request_path).parent
    data_path = Path(request["data_path"])
    try:
        if digest(data_path) != request["data_sha256"]:
            raise ValueError("Instance hash changed before execution")
        data = json.loads(data_path.read_text())
        solver = Solver(
            data,
            request["method"],
            request["seconds"],
            request["max_iterations"],
            request["witness"],
            lambda result: write(folder / "checkpoint.json", result),
        )
        result = solver.run()
        result["executed_source_sha256"] = {p.name: digest(p) for p in Path(__file__).parent.glob("*.py")}
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        result["peak_rss_mb"] = rss / (1024**2 if sys.platform == "darwin" else 1024)
        write(folder / "result.json", result)
        return 0
    except Exception as exc:
        import traceback

        write(
            folder / "error.json",
            {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()},
        )
        return 1


def run_case(data_path, metadata, method, config, folder, worker_root):
    from .data import verify
    from .solvers import relative_gap

    folder.mkdir()
    request = {
        "data_path": str(data_path),
        "data_sha256": digest(data_path),
        "method": method,
        "seconds": config.get("solver_seconds", 15.0),
        "max_iterations": config.get("max_iterations", 1000),
        "witness": metadata.get("witness"),
    }
    write(folder / "request.json", request)
    environment = os.environ.copy()
    environment.update(
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        VECLIB_MAXIMUM_THREADS="1",
        MPLBACKEND="Agg",
        PYTHONHASHSEED="0",
    )
    started = time.monotonic()
    timeout = False
    with (folder / "worker.log").open("w") as log:
        process = subprocess.Popen(
            [sys.executable, "-m", "src.pilots.lrp.run", "--worker", str(folder / "request.json")],
            cwd=worker_root,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            process.wait(timeout=request["seconds"] + 10.0)
        except subprocess.TimeoutExpired:
            timeout = True
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
    elapsed = time.monotonic() - started
    if timeout and (folder / "checkpoint.json").exists():
        result = json.loads((folder / "checkpoint.json").read_text())
        result["status"] = "feasible" if result["incumbent"] else "no_incumbent"
        result["termination_reason"] = "watchdog_timeout"
        result["checkpoint_method_seconds"] = result.pop("method_seconds")
        result["method_seconds"] = None  # the killed process did not record its final elapsed time
    elif process.returncode != 0 or not (folder / "result.json").exists():
        error = (
            json.loads((folder / "error.json").read_text())
            if (folder / "error.json").exists()
            else {"message": "Worker exited without result"}
        )
        result = {
            "method": method,
            "status": "error",
            "termination_reason": "watchdog_timeout" if timeout else "worker_error",
            "error": error,
            "incumbent": None,
            "lower_bound": None,
            "upper_bound": None,
            "relative_gap": None,
            "method_seconds": None,
            "trace": [],
            "solves": [],
            "cuts": [],
        }
    else:
        result = json.loads((folder / "result.json").read_text())
        expected_sources = {p.name: digest(p) for p in (worker_root / "src/pilots/lrp").glob("*.py")}
        if result.get("executed_source_sha256") != expected_sources:
            raise RuntimeError("Worker did not execute the frozen source snapshot")
    if digest(data_path) != request["data_sha256"]:
        raise RuntimeError("Instance modified by worker")
    data = json.loads(data_path.read_text())
    checked = verify(data, result["incumbent"]) if result.get("incumbent") else None
    if checked is not None and not checked["ok"]:
        result["status"] = "error"
        result["termination_reason"] = "independent_verification_failed"
    if result["status"] == "optimal":
        gap = relative_gap(result.get("lower_bound"), result.get("upper_bound"))
        if checked is None or gap is None or gap > 1e-7:
            result["status"] = "error"
            result["termination_reason"] = "invalid_optimality_claim"
    result.update(
        process_wall_seconds=elapsed,
        watchdog_timeout=timeout,
        worker_returncode=process.returncode,
        independent_verification=checked,
        data_sha256=request["data_sha256"],
        timing_scope="method_seconds includes validation/model setup/solves; process wall also includes interpreter imports and result serialization",
        watchdog_grace_seconds=10.0,
    )
    write(folder / "result.json", result)
    return result


def campaign(config_path, output):
    from .data import generate, validate_data
    from .report import report
    from .solvers import DEFAULT_METHODS

    config_path, output = Path(config_path).resolve(), Path(output).resolve()
    config = json.loads(config_path.read_text())
    validate_config(config)
    # Refuse output anywhere in the historical dataset, source, or published results.
    for protected in ("data", "src", "results", "configs", ".git"):
        if output.is_relative_to(ROOT / protected):
            raise ValueError(f"Output cannot be inside {protected}")
    output.mkdir(parents=True, exist_ok=False)
    write(output / "config.json", config)
    sources = sorted(Path(__file__).parent.glob("*.py"))
    snapshot = output / "source_snapshot/src/pilots/lrp"
    snapshot.mkdir(parents=True)
    # Regular packages take precedence over the editable checkout's packages.
    # Every worker runs only this snapshot, even if the live branch is edited.
    (output / "source_snapshot/src/__init__.py").write_text("")
    (output / "source_snapshot/src/pilots/__init__.py").write_text("")
    for source in sources:
        shutil.copy2(source, snapshot / source.name)
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in sources}
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
    except subprocess.CalledProcessError:
        commit, dirty = None, None
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": {p: importlib.metadata.version(p) for p in ("numpy", "scipy", "matplotlib")},
        "threads": 1,
        "backend": "scipy-highs",
        "gurobi_used": False,
    }
    try:
        from scipy.optimize._highspy import _core

        environment["highs_binary_sha256"] = digest(_core.__file__)
        environment["highs_version"] = ".".join(
            str(getattr(_core, f"HIGHS_VERSION_{part}")) for part in ("MAJOR", "MINOR", "PATCH")
        )
    except (ImportError, AttributeError):
        environment["highs_version"] = "see scipy version and native binary hash if available"
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "code_commit": commit,
        "dirty_checkout": dirty,
        "source_sha256": hashes,
        "config_sha256": digest(config_path),
        "environment": environment,
        "protocol": "lrp-local-pilot-v1",
        "instances": [],
        "run_state": "running",
        "comparison": "Same supplied route pool, one thread, equal per-method wall budgets, shared witness if generated; no solver warm start",
        "scope": "Local solver calibration, no agents; not a reproduction of the paper timings",
    }
    write(output / "manifest.json", manifest)
    rows = []
    methods = config.get("methods", list(DEFAULT_METHODS))
    for case in config["cases"]:
        for seed in case.get("seeds", [0]):
            identifier = f"{case['name']}_seed{seed:03d}"
            instance_dir = output / identifier
            instance_dir.mkdir()
            started = time.monotonic()
            if "source" in case:
                source = (ROOT / case["source"]).resolve()
                data = json.loads(source.read_text())
                validate_data(data)
                metadata = {
                    "generator": "existing-input",
                    "source": str(source),
                    "source_sha256": digest(source),
                    "route_pool": "supplied_pool_completeness_unverified",
                    "routes": len(data["routes"]),
                    "witness": None,
                }
            else:
                data, metadata = generate(case, seed)
            metadata["generation_seconds"] = time.monotonic() - started
            data_path = instance_dir / "data.json"
            write(data_path, data)
            write(instance_dir / "instance.json", metadata)
            manifest["instances"].append({"id": identifier, "data_sha256": digest(data_path), "metadata": metadata})
            write(output / "manifest.json", manifest)
            # Rotate execution order to reduce systematic first-method effects.
            offset = len(manifest["instances"]) % len(methods)
            order = methods[offset:] + methods[:offset]
            for method in order:
                print(
                    f"{identifier}: {method}, {data['m']} depots / {data['n']} customers / {len(data['routes'])} routes",
                    flush=True,
                )
                result = run_case(
                    data_path, metadata, method, config, instance_dir / method, output / "source_snapshot"
                )
                row = {
                    **result,
                    "instance": identifier,
                    "depots": data["m"],
                    "customers": data["n"],
                    "routes": len(data["routes"]),
                    "route_pool": metadata["route_pool"],
                    "generation_seconds": metadata["generation_seconds"],
                    "instance_metadata": metadata,
                }
                rows.append(row)
                write(output / "results.json", rows)
                print(
                    f"  {result['status']}: objective={result['upper_bound']}, bound={result['lower_bound']}, gap={result['relative_gap']}",
                    flush=True,
                )
    try:
        report(output, rows)
    except Exception as exc:
        manifest["run_state"] = "report_failed"
        manifest["report_error"] = {"type": type(exc).__name__, "message": str(exc)}
        write(output / "manifest.json", manifest)
        raise
    manifest["run_state"] = "complete_with_errors" if any(r["status"] == "error" for r in rows) else "complete"
    write(output / "manifest.json", manifest)
    print(f"Report: {output / 'report.md'}", flush=True)
    return int(manifest["run_state"] != "complete")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        return worker(args.worker)
    if args.config is None or args.output is None:
        parser.error("--config and --output are required")
    return campaign(args.config, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
