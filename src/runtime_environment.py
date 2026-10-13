"""Check the Python and Gurobi versions used by the experiment controller and solver."""
from __future__ import annotations

import importlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform
import sys


# These pins are copied with src into each frozen controller.
PYTHON_VERSION = "3.12.11"
GUROBI_VERSION = "11.0.3"
VERSION_FIELDS = ("python_version", "gurobipy_version", "gurobi_version")
_PYTHON_BINARY = Path(sys.executable).resolve()


class RuntimeEnvironmentError(RuntimeError):
    """The pinned execution environment is unavailable; do not score or retry a candidate."""


def require_runtime_environment(expected: dict | None = None) -> dict:
    """Inspect versions without creating a Gurobi environment or requesting a license."""
    python_version = platform.python_version()
    if python_version != PYTHON_VERSION:
        raise RuntimeEnvironmentError(f"Python {PYTHON_VERSION} is required; found {python_version} at {sys.executable}")
    binary = Path(sys.executable).resolve()
    if binary != _PYTHON_BINARY or not binary.is_file():
        raise RuntimeEnvironmentError("The Python executable changed or disappeared while the controller was running")
    try:
        package_version = version("gurobipy")
        gurobi = importlib.import_module("gurobipy")
        runtime_version = ".".join(map(str, gurobi.gurobi.version()))
    except (ImportError, OSError, PackageNotFoundError) as error:
        raise RuntimeEnvironmentError(f"Gurobi {GUROBI_VERSION} is required but could not be loaded: {error}") from error
    if package_version != GUROBI_VERSION or runtime_version != GUROBI_VERSION:
        raise RuntimeEnvironmentError(f"Gurobi {GUROBI_VERSION} is required; found gurobipy {package_version}, runtime {runtime_version}")
    observed = {"python_version": python_version, "python_executable": sys.executable, "python_binary": str(binary),
                "gurobipy_version": package_version, "gurobi_version": runtime_version}
    if expected is not None and any(observed[key] != expected.get(key) for key in VERSION_FIELDS):
        raise RuntimeEnvironmentError("Python/Gurobi versions differ from the frozen campaign environment")
    return observed


if __name__ == "__main__":
    try:
        print(json.dumps(require_runtime_environment(), indent=2))
    except RuntimeEnvironmentError as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(64) from error
