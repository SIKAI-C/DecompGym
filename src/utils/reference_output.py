"""Convert reference solver metadata into the agent's standard output contract."""
from __future__ import annotations

from collections import defaultdict
from typing import Any


def reference_output(suite: str, problem: str, data: dict, result: dict) -> dict[str, Any]:
    output = {"status": "feasible", "method": "benders_decomposition" if suite == "bd" else "column_generation", "objective_value": result["best_objective"]}
    metadata = {**result.get("master_metadata", {}), **result.get("sub_metadata", {})}
    variables = result.get("final_vars", {})
    if suite == "bd":
        if problem == "tsso":
            output["y"] = metadata["y_assignment"]
            recourse = metadata["recourse"]
            for name in ["x_reg", "x_ot", "inv", "back"]:
                if name in recourse:
                    output[name] = recourse[name]
                else:
                    output[name] = [[[(recourse.get(str(s)) or recourse[s])[name][i][t] for s in range(data["n_scenarios"])] for t in range(data["n_periods"])] for i in range(data["n_products"])]
        else:
            output["y"] = metadata.get("y_assignment")
            if output["y"] is None:
                output["y"] = [variables[f"y_{i}"] for i in range(data["m"])]
            if problem == "lrp":
                output["lambda"] = metadata.get("lambda_assignment_int", metadata["lambda_assignment"])
            else:
                output["x"] = metadata["x_assignment"]
    elif problem == "bp":
        # Reference patterns aggregate identical sizes. Map them back to individual item IDs and trim surplus.
        groups = defaultdict(list)
        for item in sorted(data["items"], key=lambda item: item["id"], reverse=True):
            groups[item["size"]].append(item["id"])
        sizes = sorted(groups)
        bins = []
        for pattern in metadata.get("solution_patterns", metadata["patterns"]):
            for _ in range(round(pattern["usage"])):
                items = []
                for entry in pattern["contents"]:
                    available = groups[sizes[entry["item_id"]]]
                    for _ in range(min(round(entry["count"]), len(available))):
                        items.append({"item_id": available.pop(), "count": 1})
                if items:
                    bins.append({"usage": 1, "contents": items})
        output["solution"] = {"bins": bins}
    elif problem == "csp":
        output["solution"] = {"patterns": metadata.get("solution_patterns", metadata["patterns"])}
    elif problem == "gap":
        output["solution"] = {"assignments": metadata["assignments"]}
    elif problem == "shift":
        output["solution"] = {"shifts": metadata["shifts"]}
    return output
