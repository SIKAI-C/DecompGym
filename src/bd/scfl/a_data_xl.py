"""Two-stage stochastic CFL at scale: the pinned instance generator of the Stage 1 / holdout panels (protocol v4
sections 14-15). Self-contained (NumPy + JSON only) so that later edits of the shared data module
src/bd/scfl/a_data.py cannot silently change a frozen instance; tests/large_scale/test_bdlp_stage1.py checks that both
still produce byte-identical files, and every generated file's sha256 is frozen in the panel.

Recipe (Bodur, Dash, Guenluek and Luedtke 2017, INFORMS JoC 29(1):77-91, Section 4.4, on the OR-Library CAP
instances of Beasley 1990, http://people.brunel.ac.uk/~mastjjb/jeb/orlib/capinfo.html):
  - f_i, u_i, mean demands dbar_j and the cost of serving ALL of customer j from facility i are read from the CAP
    file; the unit cost is c_ij = cost_ij / dbar_j;
  - per customer sigma_j ~ U(0.1 dbar_j, 0.3 dbar_j) (drawn once), then d_sj ~ N(dbar_j, sigma_j) independently for
    every scenario, equal probabilities 1/S.
Our documented choices where the paper is silent (as in a_data.make_stochastic): demands rounded to the nearest
integer (numpy rint, half to even) and truncated at 0; numpy default_rng(seed), sigma first, then the S x J demand
matrix in row order; numbers written as ints when integral (else repr floats); name <base>_s<S>_g<seed>.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

SIGMA = (0.1, 0.3)


def read_cap(path: str | Path) -> dict:
    """OR-Library capacitated warehouse location file: 'm n', m lines 'capacity fixed_cost', then per customer its
    demand followed by the m costs of serving all of its demand from each warehouse."""
    tok = Path(path).read_text().split()
    m, n = int(tok[0]), int(tok[1])
    if len(tok) != 2 + 2 * m + n * (m + 1):
        raise ValueError(f"{path}: {len(tok)} tokens, expected {2 + 2 * m + n * (m + 1)} for m={m}, n={n}")
    vals = [float(t) for t in tok[2:]]
    cap = np.array(vals[0:2 * m:2])
    fixed = np.array(vals[1:2 * m:2])
    rest = np.array(vals[2 * m:]).reshape(n, m + 1)
    demand = rest[:, 0]
    if np.any(demand <= 0):
        raise ValueError("OR-Library customers must have positive demand")
    return {"capacity": cap, "fixed": fixed, "demand": demand, "alloc_cost": rest[:, 1:].T, "name": Path(path).stem}


def _num(v: float):
    v = float(v)
    return int(v) if v.is_integer() and abs(v) < 2 ** 53 else v


def make_instance(base: dict, scenarios: int, seed: int) -> dict:
    """The scfl-v1 instance object (module docstring); raises if the total capacity is below a scenario demand."""
    rng = np.random.default_rng(seed)
    dbar = base["demand"]
    sig = rng.uniform(SIGMA[0] * dbar, SIGMA[1] * dbar)
    raw = rng.normal(dbar[None, :], sig[None, :], size=(scenarios, len(dbar)))
    demand = np.maximum(np.rint(raw), 0.0)
    cost = base["alloc_cost"] / dbar[None, :]
    if float(base["capacity"].sum()) < float(demand.sum(axis=1).max()):
        raise ValueError("generated instance is infeasible (total capacity below a scenario demand)")
    meta = {"base": base["name"], "scenarios": scenarios, "seed": seed, "sigma": list(SIGMA),
            "generator": "Bodur et al. (2017) Sec. 4.4: sigma_j ~ U(0.1, 0.3) dbar_j, d_sj ~ N(dbar_j, sigma_j); "
                         "rounded to the nearest integer, truncated at 0; numpy default_rng(seed)",
            "truncated_values": int(np.sum(raw < -0.5))}
    return {"format": "scfl-v1", "name": f"{base['name']}_s{scenarios}_g{seed}",
            "capacity": [_num(v) for v in base["capacity"]], "fixed_cost": [_num(v) for v in base["fixed"]],
            "unit_cost": [[_num(v) for v in row] for row in cost], "demand": [[_num(v) for v in row] for row in demand],
            "meta": meta}


def write_instance(obj: dict, path: str | Path) -> str:
    """Compact JSON; returns the sha256 of the written bytes."""
    text = json.dumps(obj, separators=(",", ":"))
    Path(path).write_text(text)
    return hashlib.sha256(text.encode()).hexdigest()
