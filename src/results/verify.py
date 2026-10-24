"""Independently recalculate the plotted outcomes and printed table cells from the raw policy summaries."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re

from src.results.paper_figures import MODES, TABLE_BUDGETS, digest
from src.results.policy import PAPER_MODELS


def verify(out):
    data = json.loads((out / "combined.json").read_text())
    models = set(data["primary_models"])
    assert models == set(PAPER_MODELS)
    recomputed = {}
    source_hashes = {}
    for pair in data["pairs"]:
        if pair["model"] not in models:
            continue
        for mode in MODES:
            policy = pair["policies"][mode]
            path = Path(pair["input_root"]) / policy["path"] / "instance_summary.json"
            raw = path.read_bytes()
            summary = json.loads(raw)
            assert summary["complete"] and summary["model"] == pair["model"]
            assert summary["max_attempts"] == 10
            source_hashes[str(path)] = hashlib.sha256(raw).hexdigest()
            gold = policy["gold_objective"]
            for k in range(1, 11):
                prefix = [a for a in summary["attempts"] if a["attempt"] <= k]
                scores = {
                    "execution": any(a["exec_ok"] for a in prefix),
                    "verification": any(a["verify_ok"] for a in prefix),
                    "quality": any(a["verify_ok"] and isinstance(a.get("objective"), (float, int)) and math.isfinite(a["objective"]) and abs(a["objective"] - gold) / max(1, abs(gold)) <= .01 for a in prefix),
                }
                assert scores == policy["attempt_metrics"][str(k)]
                assert scores["quality"] <= scores["verification"] <= scores["execution"]
                recomputed[(pair["model"], pair["instance"], mode, k)] = scores
    assert len(source_hashes) == 1944
    assert len(recomputed) == 19440
    pairs = [p for p in data["pairs"] if p["model"] in models]
    for p in pairs:
        assert recomputed[(p["model"], p["instance"], "plain", 1)] == recomputed[(p["model"], p["instance"], "reflection", 1)]
    assert Counter(p["size"] for p in pairs) == {"small": 324, "medium": 324, "large": 324}
    curves_checked = 0
    for row in data["rows"]:
        if row["model"] not in models:
            continue
        own = [p for p in pairs if p["model"] == row["model"] and (row["dimension"] == "overall" or (p["suite"]+"/"+p["problem"] if row["dimension"] == "problem" else p[row["dimension"]]) == row["group"])]
        assert len(own) == row["paired_cases"]
        for mode in MODES:
            count = sum(recomputed[(p["model"], p["instance"], mode, row["budget"])][row["metric"]] for p in own)
            assert count == row[mode+"_successes"]
            assert math.isclose(count/len(own), row[mode+"_rate"], abs_tol=1e-12)
        curves_checked += 1
    aggregate = {str(k): {metric: {mode: sum(recomputed[(p["model"], p["instance"], mode, k)][metric] for p in pairs) for mode in MODES} for metric in ("execution", "verification", "quality")} for k in (1, 2, 3, 10)}
    assert [aggregate[str(k)]["quality"][mode] for k, mode in ((1,"plain"),(3,"plain"),(3,"reflection"),(10,"plain"),(10,"reflection"))] == [853,936,963,963,970]
    assert [aggregate["2"]["quality"][mode] for mode in MODES] == [920,947]
    # Verify printed table percentages against raw summaries, independently of the table writer.
    printed = [re.findall(r"(\d+\.\d+)\\%", line) for line in (out/"tables/main_results.tex").read_text().splitlines() if r"\%" in line]
    assert len(printed) == 13
    table_cells_checked = 0
    for index, model in enumerate([*data["primary_models"], None]):
        own = [p for p in pairs if model is None or p["model"] == model]
        assert len(printed[index]) == len(TABLE_BUDGETS)
        for value, (k, mode) in zip(printed[index], TABLE_BUDGETS):
            count = sum(recomputed[(p["model"], p["instance"], mode, k)]["quality"] for p in own)
            assert value == f"{100*count/len(own):.1f}"
            table_cells_checked += 1
    report = {"verified_at": datetime.now(timezone.utc).isoformat(), "status": "pass", "physical_policy_summaries_read": len(source_hashes), "candidate_budget_states_checked": len(recomputed), "curve_rows_checked": curves_checked, "model_count": 12, "pair_count": 972, "shared_first_equal": True, "aggregate_counts": aggregate, "raw_summary_hashes": source_hashes, "combined_sha256": digest(out/"combined.json"), "verification_code_sha256": digest(__file__)}
    report["printed_percentage_cells_checked"] = table_cells_checked
    (out/"qa/raw_outcome_verification.json").write_text(json.dumps(report,indent=2)+"\n")
    return {k:v for k,v in report.items() if k != "raw_summary_hashes"}
