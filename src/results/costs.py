"""Join per-policy time and token rows and summarize them by model, mode and group."""
from __future__ import annotations

from collections import defaultdict

import numpy as np


PAIR_KEY = ("cohort_protocol", "model", "instance", "mode")


def key(row):
    return tuple(row[k] for k in PAIR_KEY)


def combine_costs(time, tokens):
    token_index = {key(r): r for r in tokens["case_policies"]}
    if len(token_index) != len(tokens["case_policies"]):
        raise ValueError("Duplicate token policy")
    rows = []
    for row in time["case_policies"]:
        other = token_index.pop(key(row))
        if row["quality"] != other["quality"] or row["first_quality_attempt"] != other["first_quality_attempt"]:
            raise ValueError("Time and token outcomes differ")
        rows.append(row | {k: v for k, v in other.items() if k not in row})
    if token_index:
        raise ValueError("Time/token policy populations differ")
    return rows


def mean(rows, field):
    values = [r[field] for r in rows]
    return float(np.mean(values)) if values and all(v is not None for v in values) else None


def summaries(rows):
    grouped = defaultdict(list)
    for row in rows:
        for dimension, group in (("overall", "all"), ("suite", row["suite"]), ("problem", row["suite"] + "/" + row["problem"]), ("size", row["size"])):
            grouped[(row["model"], row["mode"], dimension, group)].append(row)
    result = []
    for (model, mode, dimension, group), cases in grouped.items():
        successes = sum(r["quality"] for r in cases)
        entry = {"model": model, "mode": mode, "dimension": dimension, "group": group, "primary_model": cases[0]["primary_model"], "cases": len(cases), "quality_successes": successes, "quality_rate": successes / len(cases)}
        for field in ("agent_seconds", "active_work_seconds", "judge_seconds", "solver_seconds", "generation_seconds", "reflection_seconds", "agent_accounted_tokens", "agent_context_input_tokens", "agent_output_tokens", "agent_reasoning_tokens", "agent_cache_read_tokens", "agent_cache_creation_tokens", "judge_accounted_tokens"):
            entry["mean_" + field] = mean(cases, field)
            available = [r[field] for r in cases if r[field] is not None]
            entry["available_" + field] = len(available)
            if field in ("agent_seconds", "agent_accounted_tokens"):
                entry["median_" + field] = float(np.median(available)) if available else None
                entry["p90_" + field] = float(np.quantile(available, .9)) if available else None
                entry["per_observed_success_" + field] = sum(available) / successes if successes and len(available) == len(cases) else None
        result.append(entry)
    return result
