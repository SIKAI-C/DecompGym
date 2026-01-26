"""
Basic verifier for TSSO solutions emitted per the prompt contract.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _to_float(val: Any):
    try:
        return float(val)
    except Exception:
        return None


def _check_shape(arr: list, shape: tuple[int, ...]) -> bool:
    try:
        if len(arr) != shape[0]:
            return False
        if len(shape) == 1:
            return True
        inner = shape[1:]
        return all(_check_shape(sub, inner) for sub in arr)
    except Exception:
        return False


def verify_solution(data_path: str | Path, output_path: str | Path, tol: float = 1e-6) -> dict[str, Any]:
    data = json.loads(Path(data_path).read_text())
    out = json.loads(Path(output_path).read_text())

    n_products = int(data.get("n_products", 0))
    n_periods = int(data.get("n_periods", 0))
    n_scenarios = int(data.get("n_scenarios", 0))

    y = out.get("y")
    x_reg = out.get("x_reg")
    x_ot = out.get("x_ot")
    inv = out.get("inv")
    back = out.get("back")

    issues: list[str] = []

    if y is None or not _check_shape(y, (n_products, n_periods)):
        issues.append("y missing or has incorrect shape")
    if x_reg is None or not _check_shape(x_reg, (n_products, n_periods, n_scenarios)):
        issues.append("x_reg missing or has incorrect shape")
    if x_ot is None or not _check_shape(x_ot, (n_products, n_periods, n_scenarios)):
        issues.append("x_ot missing or has incorrect shape")
    if inv is None or not _check_shape(inv, (n_products, n_periods, n_scenarios)):
        issues.append("inv missing or has incorrect shape")
    if back is None or not _check_shape(back, (n_products, n_periods, n_scenarios)):
        issues.append("back missing or has incorrect shape")
    if issues:
        return {"ok": False, "issues": issues}

    for row in y:
        for val in row:
            fval = _to_float(val)
            if fval is None:
                issues.append(f"y entry {val} not numeric")
                break
            if not (0.0 - tol <= fval <= 1.0 + tol):
                issues.append(f"y entry {val} not in [0,1]")
                break

    demand = data.get("demand", [])
    K = data.get("K", [])
    I0 = data.get("I0", [0.0] * n_products)
    B0 = data.get("B0", [0.0] * n_products)

    for s in range(n_scenarios):
        for i in range(n_products):
            for t in range(n_periods):
                cap = _to_float(K[i][t])
                y_it = _to_float(y[i][t])
                x_reg_val = _to_float(x_reg[i][t][s])
                x_ot_val = _to_float(x_ot[i][t][s])
                inv_val = _to_float(inv[i][t][s])
                back_val = _to_float(back[i][t][s])
                dem_val = _to_float(demand[i][t][s])
                inv_prev_val = _to_float(I0[i]) if t == 0 else _to_float(inv[i][t - 1][s])
                back_prev_val = _to_float(B0[i]) if t == 0 else _to_float(back[i][t - 1][s])

                if None in (cap, y_it, x_reg_val, x_ot_val, inv_val, back_val, dem_val, inv_prev_val, back_prev_val):
                    issues.append(f"scenario {s}: non-numeric value at product {i}, period {t}")
                    continue

                lhs = x_reg_val
                if lhs - cap * y_it > tol:
                    issues.append(f"scenario {s}: capacity violated at product {i}, period {t}")
                inv_prev = inv_prev_val
                back_prev = back_prev_val
                balance = x_reg_val + x_ot_val - inv_val + back_val - (dem_val - inv_prev + back_prev)
                if abs(balance) > tol:
                    issues.append(f"scenario {s}: balance mismatch at product {i}, period {t}")

    return {"ok": len(issues) == 0, "issues": issues}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Verify a TSSO output.json against basic constraints.")
    parser.add_argument("--data", required=True, help="Path to data.json")
    parser.add_argument("--output", required=True, help="Path to output.json")
    parser.add_argument("--tol", type=float, default=1e-6, help="Tolerance for equality/inequality checks")
    args = parser.parse_args()

    res = verify_solution(args.data, args.output, tol=args.tol)
    if res["ok"]:
        print("OK: solution satisfies basic constraints within tolerance.")
    else:
        print("INVALID:")
        for msg in res["issues"]:
            print(f" - {msg}")
