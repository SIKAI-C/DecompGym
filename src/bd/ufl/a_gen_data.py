from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from utils.base_data import BaseInstanceGenerator, JsonDataMixin
import matplotlib.pyplot as plt
import numpy as np


# -----------------------
# Instance generation
# -----------------------

DIFFICULTY_CONFIG = {
    "easy": {
        "m_range": (5, 10),   # facilities
        "n_range": (15, 30),   # customers
    },
    "medium": {
        "m_range": (10, 20),
        "n_range": (30, 60),
    },
    "hard": {
        "m_range": (20, 40),
        "n_range": (60, 120),
    },
}


@dataclass
class UFLInstance(JsonDataMixin):
    """Container for an Uncapacitated Facility Location instance."""

    m: int  # number of facilities
    n: int  # number of customers
    Xc: np.ndarray  # (n, 2) customer coords
    Xf: np.ndarray  # (m, 2) facility coords
    f: np.ndarray  # (m,) fixed opening costs
    D: np.ndarray  # (m, n) shipping costs

    def __post_init__(self) -> None:
        self.Xc = np.asarray(self.Xc, dtype=float)
        self.Xf = np.asarray(self.Xf, dtype=float)
        self.f = np.asarray(self.f, dtype=float)
        self.D = np.asarray(self.D, dtype=float)

    @classmethod
    def from_json_file(cls, path: str | Path) -> UFLInstance:
        # Narrow return type for convenience
        return cast("UFLInstance", super().from_json_file(path))


class UFLGenerator(BaseInstanceGenerator[UFLInstance]):
    """Synthetic UFL generator mirroring the tutorial recipe."""

    def __init__(self, *, seed: int = 1804) -> None:
        self.default_seed = seed

    def generate_instance(
        self,
        m: int = 20,
        n: int = 100,
        *,
        seed: int | None = None,
        grid_param: int = 1024,
    ) -> UFLInstance:
        rng = np.random.default_rng(seed if seed is not None else self.default_seed)

        Xc = rng.integers(0, grid_param + 1, size=(n, 2)) / float(grid_param)
        Xf = rng.integers(-1, 22, size=(m, 2)) / 20.0
        f = rng.integers(10, 21, size=m).astype(float)

        diff = Xf[:, None, :] - Xc[None, :, :]
        D = np.linalg.norm(diff, axis=2) * 10.0

        return UFLInstance(m=m, n=n, Xc=Xc, Xf=Xf, f=f, D=D)

    def _ensure_ax(self, ax: plt.Axes | None) -> plt.Axes:
        if ax is not None:
            return ax
        _, ax_new = plt.subplots()
        return ax_new

    def plot_instance(self, inst: UFLInstance, output_path: str | Path | None = None) -> None:
        ax = self._ensure_ax(None)
        ax.set_title("UFL instance")
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.33)

        ax.scatter(inst.Xf[:, 0], inst.Xf[:, 1], marker="s", s=70, label="Facilities", alpha=0.6)
        ax.scatter(inst.Xc[:, 0], inst.Xc[:, 1], marker="o", s=40, label="Customers", color="tab:red")
        ax.legend(loc="upper left")

        if output_path is None:
            plt.show()
        else:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, format="pdf", bbox_inches="tight")
            plt.close(ax.figure)

    def plot_solution(
        self,
        inst: UFLInstance,
        result: dict[str, Any],
        *,
        output_path: str | Path | None = None,
        open_tol: float = 0.5,
    ) -> None:
        """Plot using the finalized result dict from baseline or Benders."""
        vars_dict = result.get("final_vars", {}) or {}
        master_meta = result.get("master_metadata", {}) or {}
        sub_meta = result.get("sub_metadata", {}) or {}

        if "x_assignment" in sub_meta:
            x_arr = np.asarray(sub_meta["x_assignment"], dtype=float).reshape(inst.m, inst.n)
        elif "x_assignment" in master_meta:
            x_arr = np.asarray(master_meta["x_assignment"], dtype=float).reshape(inst.m, inst.n)
        else:
            x_arr = np.zeros((inst.m, inst.n), dtype=float)
            for i in range(inst.m):
                for j in range(inst.n):
                    x_arr[i, j] = float(vars_dict.get(f"x_{i}_{j}", 0.0) or 0.0)

        y_arr = np.asarray([float(vars_dict.get(f"y_{i}", 0.0) or 0.0) for i in range(inst.m)], dtype=float)

        ax = self._ensure_ax(None)
        ax.set_title("UFL solution")
        ax.set_ylim(-0.1, 1.33)

        for i in range(inst.m):
            for j in range(inst.n):
                if x_arr[i, j] > 1e-6:
                    ax.plot(
                        [inst.Xc[j, 0], inst.Xf[i, 0]],
                        [inst.Xc[j, 1], inst.Xf[i, 1]],
                        color="black",
                        linewidth=0.5,
                        alpha=min(0.95, 0.25 + x_arr[i, j]),
                    )

        open_mask = y_arr >= open_tol
        ax.scatter(
            inst.Xf[~open_mask, 0],
            inst.Xf[~open_mask, 1],
            marker="s",
            s=60,
            color="lightblue",
            label="Facilities (closed)",
        )
        ax.scatter(
            inst.Xf[open_mask, 0],
            inst.Xf[open_mask, 1],
            marker="s",
            s=60,
            color="darkblue",
            label="Facilities (open)",
        )
        ax.scatter(
            inst.Xc[:, 0],
            inst.Xc[:, 1],
            marker="o",
            s=35,
            color="tab:red",
            label="Customers",
        )
        ax.legend(loc="upper left")

        if output_path is None:
            plt.show()
        else:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, format="pdf", bbox_inches="tight")
            plt.close(ax.figure)


def generate_ufl_instance(
    seed: int,
    difficulty: str = "easy",
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """
    Sample (m, n) from difficulty-specific ranges, generate a UFL instance,
    optionally write it to output_dir/data.json, and return the data dict.
    """
    if difficulty not in DIFFICULTY_CONFIG:
        raise ValueError(f"Unknown difficulty '{difficulty}'. Expected one of {list(DIFFICULTY_CONFIG)}.")

    cfg = DIFFICULTY_CONFIG[difficulty]
    rng = np.random.default_rng(seed)

    m = int(rng.integers(cfg["m_range"][0], cfg["m_range"][1] + 1))
    n = int(rng.integers(cfg["n_range"][0], cfg["n_range"][1] + 1))

    gen = UFLGenerator(seed=seed)
    inst = gen.generate_instance(m=m, n=n, seed=seed)
    data = inst.to_dict()

    if output_dir is not None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        inst.to_json_file(out_dir / "data.json")

    return data
