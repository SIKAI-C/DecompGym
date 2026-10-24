"""A plain subprocess solver runner for tests; real campaigns use the macOS sandbox executor."""
import json
import os
from pathlib import Path
import subprocess
import sys


def host_solver(attempt_dir: Path, timeout: float, threads: int = 1):
    if not (attempt_dir / "solver.py").exists() or not (attempt_dir / "solver.py").stat().st_size:
        return False, "", "solver.py missing or empty", None
    (attempt_dir / "output.json").unlink(missing_ok=True)
    env = os.environ | {"OMP_NUM_THREADS": str(threads)}
    try:
        process = subprocess.run([sys.executable, "solver.py"], cwd=attempt_dir, capture_output=True, text=True, env=env, timeout=timeout)
    except subprocess.TimeoutExpired as error:
        return False, error.stdout or "", (error.stderr or "") + f"\n[framework] solver execution timed out after {timeout} seconds", None
    if process.returncode:
        return False, process.stdout, process.stderr or f"returncode={process.returncode}", None
    try:
        output = json.loads((attempt_dir / "output.json").read_text())
    except (OSError, ValueError):
        output = None
    if not isinstance(output, dict):
        return False, process.stdout, process.stderr + "\noutput.json missing, invalid, or not an object", None
    return True, process.stdout, process.stderr, output
