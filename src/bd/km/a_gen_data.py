from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np

from src.utils.base_data import BaseInstanceGenerator, JsonDataMixin


DIFFICULTY_CONFIG = {
    "easy": {"m_range": (5, 10), "n_range": (10, 20), "k_frac": (0.2, 0.3)},
    "medium": {"m_range": (10, 15), "n_range": (20, 30), "k_frac": (0.2, 0.35)},
    "hard": {"m_range": (15, 20), "n_range": (30, 40), "k_frac": (0.15, 0.35)},
}


@dataclass
class KMedianInstance(JsonDataMixin):
    """k-median instance data."""

    m: int  # facilities
    n: int  # customers
    k: int  # facilities to open
    Xc: np.ndarray  # (n, 2)
    Xf: np.ndarray  # (m, 2)
    D: np.ndarray  # (m, n) assignment cost

    def __post_init__(self) -> None:
        self.Xc = np.asarray(self.Xc, dtype=float)
        self.Xf = np.asarray(self.Xf, dtype=float)
        self.D = np.asarray(self.D, dtype=float)
        if not (0 < self.k <= self.m):
            raise ValueError("Require 0 < k <= m.")
        if self.Xc.shape != (self.n, 2):
            raise ValueError("Xc shape mismatch")
        if self.Xf.shape != (self.m, 2):
            raise ValueError("Xf shape mismatch")
        if self.D.shape != (self.m, self.n):
            raise ValueError("D shape mismatch")

    @classmethod
    def from_json_file(cls, path: str | Path) -> KMedianInstance:
        return cast("KMedianInstance", super().from_json_file(path))


class KMedianGenerator(BaseInstanceGenerator[KMedianInstance]):
    """Synthetic k-median generator matching the tutorial recipe."""

    def __init__(self, *, seed: int = 1804, grid_param: int = 1024) -> None:
        self.default_seed = seed
        self.grid_param = grid_param

    def generate_instance(
        self,
        m: int = 20,
        n: int = 100,
        k: int = 5,
        *,
        seed: int | None = None,
    ) -> KMedianInstance:
        if k <= 0 or k > m:
            raise ValueError("Require 0 < k <= m.")

        rng = np.random.default_rng(seed if seed is not None else self.default_seed)

        Xc = rng.integers(0, self.grid_param + 1, size=(n, 2)) / float(self.grid_param)
        Xf = rng.integers(-1, 22, size=(m, 2)) / 20.0

        diff = Xf[:, None, :] - Xc[None, :, :]
        D = np.linalg.norm(diff, axis=2) * 10.0

        return KMedianInstance(m=m, n=n, k=k, Xc=Xc, Xf=Xf, D=D)

    def _ensure_fig_ax(self) -> tuple[plt.Figure, plt.Axes]:
        fig, ax = plt.subplots()
        return fig, ax

    def plot_instance(self, inst: KMedianInstance, output_path: str | Path | None = None) -> None:
        fig, ax = self._ensure_fig_ax()
        ax.set_title("k-median instance")
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.33)
        ax.scatter(inst.Xf[:, 0], inst.Xf[:, 1], marker="s", s=70, label="Facilities", alpha=0.6)
        ax.scatter(inst.Xc[:, 0], inst.Xc[:, 1], marker="o", s=40, color="tab:red", label="Customers")
        ax.legend(loc="upper left")

        if output_path is None:
            plt.show()
        else:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, format="pdf", bbox_inches="tight")
            plt.close(fig)

    def plot_solution(
        self,
        inst: KMedianInstance,
        result: dict[str, Any],
        *,
        output_path: str | Path | None = None,
        open_tol: float = 0.5,
    ) -> None:
        """Plot using finalized result dict from baseline or Benders."""
        vars_dict = result.get("final_vars", {}) or {}
        sub_meta = result.get("sub_metadata", {}) or {}
        master_meta = result.get("master_metadata", {}) or {}

        if "x_assignment" in sub_meta:
            x_arr = np.asarray(sub_meta["x_assignment"], dtype=float).reshape(inst.m, inst.n)
        elif "x_assignment" in master_meta:
            x_arr = np.asarray(master_meta["x_assignment"], dtype=float).reshape(inst.m, inst.n)
        else:
            x_arr = np.zeros((inst.m, inst.n), dtype=float)
            for i in range(inst.m):
                for j in range(inst.n):
                    x_arr[i, j] = float(vars_dict.get(f"x_{i}_{j}", 0.0) or 0.0)

        if "y_assignment" in sub_meta:
            y_arr = np.asarray(sub_meta["y_assignment"], dtype=float).reshape(inst.m)
        elif "y_assignment" in master_meta:
            y_arr = np.asarray(master_meta["y_assignment"], dtype=float).reshape(inst.m)
        else:
            y_arr = np.array([vars_dict.get(f"y_{i}", 0.0) or 0.0 for i in range(inst.m)], dtype=float)

        fig, ax = self._ensure_fig_ax()
        ax.set_title(f"k-median solution (k={inst.k})")
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
        ax.scatter(inst.Xf[~open_mask, 0], inst.Xf[~open_mask, 1], marker="s", s=60, color="lightblue", label="Closed")
        ax.scatter(inst.Xf[open_mask, 0], inst.Xf[open_mask, 1], marker="s", s=60, color="darkblue", label="Open")
        ax.scatter(inst.Xc[:, 0], inst.Xc[:, 1], marker="o", s=35, color="tab:red", label="Customers")
        ax.legend(loc="upper left")

        if output_path is None:
            plt.show()
        else:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, format="pdf", bbox_inches="tight")
            plt.close(fig)


def generate_km_instance(
    seed: int,
    difficulty: str = "easy",
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    if difficulty not in DIFFICULTY_CONFIG:
        raise ValueError(f"Unknown difficulty '{difficulty}'. Expected one of {list(DIFFICULTY_CONFIG)}.")
    cfg = DIFFICULTY_CONFIG[difficulty]
    rng = np.random.default_rng(seed)
    m = int(rng.integers(cfg["m_range"][0], cfg["m_range"][1] + 1))
    n = int(rng.integers(cfg["n_range"][0], cfg["n_range"][1] + 1))
    frac = rng.uniform(cfg["k_frac"][0], cfg["k_frac"][1])
    k = max(1, min(m - 1, round(frac * m)))

    gen = KMedianGenerator(seed=seed)
    inst = gen.generate_instance(m=m, n=n, k=k, seed=seed)
    data = inst.to_dict()

    if output_dir is not None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        inst.to_json_file(out_dir / "data.json")

    return data
