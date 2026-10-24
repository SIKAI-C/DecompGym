"""Read-only token accounting for audited DecompBench snapshots."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from src.results.policy import PAPER_MODELS


AGENT_STAGES = {"generation", "reflection"}
TOKEN_BUDGETS = (25000, 50000, 100000, 150000, 250000, 500000, 1000000, 2000000, 5000000, 10000000)
COMPONENTS = ("context_input_tokens", "output_tokens", "accounted_tokens", "reasoning_tokens", "cache_read_tokens", "cache_creation_tokens", "provider_reported_total_tokens")


def integer(value):
    return value if type(value) is int and value >= 0 else None


def strict_sum(values):
    values = list(values)
    return sum(values) if all(v is not None for v in values) else None


def supplemental_claude(call, input_root):
    """Read only numeric terminal metadata, retaining a file hash and consistency check."""
    stage = {"generation": "agent", "reflection": "reflection"}.get(call["stage"])
    if call["backend"] != "claude-code-cli" or stage is None:
        return {}
    root = Path(input_root).resolve()
    path = root / call["path"] / (stage + "_events.jsonl")
    if not path.is_file():
        return {"supplement_state": "missing_events"}
    if path.is_symlink() or not path.resolve().is_relative_to(root):
        raise ValueError("Supplemental evidence leaves the read-only source root")
    before = path.stat()
    content = path.read_bytes()
    after = path.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
        return {"supplement_state": "changed_during_read"}
    terminal = []
    for line in content.splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if isinstance(event, dict) and event.get("type") == "result":
            terminal.append(event)
    if len(terminal) != 1:
        return {"supplement_state": "terminal_result_not_unique", "terminal_results": len(terminal)}
    event = terminal[0]
    raw_usage = event.get("usage") or {}
    projection = call.get("usage") or {}
    if any(raw_usage.get(key) != value for key, value in projection.items()):
        raise ValueError(f"Supplemental usage differs from audited projection: {call['invocation_id']}")
    thinking = integer((raw_usage.get("output_tokens_details") or {}).get("thinking_tokens"))
    if thinking is not None and integer(raw_usage.get("output_tokens")) is not None and thinking > raw_usage["output_tokens"]:
        raise ValueError("Thinking tokens exceed output tokens")
    model_thinking = [integer(v.get("thinkingTokens")) for v in (event.get("modelUsage") or {}).values() if isinstance(v, dict)]
    if thinking is not None and model_thinking and all(v is not None for v in model_thinking) and sum(model_thinking) != thinking:
        raise ValueError("Claude terminal thinking differs from modelUsage")
    result = {"supplement_state": "verified", "thinking_tokens": thinking, "provider_duration_seconds": event["duration_ms"] / 1000 if integer(event.get("duration_ms")) is not None else None, "provider_api_seconds": event["duration_api_ms"] / 1000 if integer(event.get("duration_api_ms")) is not None else None, "num_turns": integer(event.get("num_turns")), "evidence_path": str(path), "evidence_sha256": hashlib.sha256(content).hexdigest()}
    return result


def token_components(call, supplemental=None):
    """Cache/reasoning fields are components, not additional tokens to add twice."""
    usage = call.get("usage") if call.get("usage_available") else None
    usage = usage if isinstance(usage, dict) else {}
    raw = {key: integer(value) for key, value in usage.items()}
    backend = call["backend"]
    output = raw.get("output_tokens")
    if backend == "codex-cli":
        context = raw.get("input_tokens")
        cache = raw.get("cached_input_tokens")
        creation = raw.get("cache_write_input_tokens")
        reasoning = raw.get("reasoning_output_tokens")
    elif backend == "claude-code-cli":
        context = strict_sum(raw.get(k) for k in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"))
        cache = raw.get("cache_read_input_tokens")
        creation = raw.get("cache_creation_input_tokens")
        reasoning = integer((supplemental or {}).get("thinking_tokens"))
    else:
        # AGY cache counts can exceed input; the CLI's raw total is not a proven full-context total.
        context = None
        cache = raw.get("cache_read_tokens")
        creation = None
        reasoning = raw.get("thinking_tokens")
    if reasoning is not None and output is not None and reasoning > output:
        raise ValueError("Reasoning/thinking tokens exceed output tokens")
    if backend == "codex-cli" and cache is not None and context is not None and cache > context:
        raise ValueError("Codex cached input exceeds total input")
    return {"context_input_tokens": context, "output_tokens": output, "accounted_tokens": strict_sum((context, output)), "reasoning_tokens": reasoning, "cache_read_tokens": cache, "cache_creation_tokens": creation, "provider_reported_total_tokens": raw.get("total_tokens")}


def statistics(values):
    available = [v for v in values if v is not None]
    return {"cases": len(values), "available_cases": len(available), "missing_cases": len(values) - len(available), "total": sum(available) if len(available) == len(values) else None, "observed_total": sum(available), **{key: float(fn(available)) if available else None for key, fn in (("mean", np.mean), ("median", np.median), ("p90", lambda a: np.quantile(a, 0.9)), ("max", max))}}


def groups(row):
    return (("overall", "all"), ("suite", row["suite"]), ("problem", row["suite"] + "/" + row["problem"]), ("size", row["size"]))


def build_token_analysis(reports, *, read_supplemental=True):
    calls, supplements, policies, budgets, coverage = [], [], [], [], []
    global_ids = set()
    for report in reports:
        if report["audit_errors"] or report["snapshot"]["read_errors"]:
            raise ValueError("Token exploration requires clean canonical audits")
        protocol = report["configuration"]["protocol_id"]
        pairs = [p for p in report["pairs"] if p["model"] in PAPER_MODELS]
        inventory = {(c["suite"], c["problem"], c["instance"]) for c in report["cohort_manifest"]["instances"]}
        primary = set()
        for model in dict.fromkeys(p["model"] for p in pairs):
            done = [p for p in pairs if p["model"] == model and p["state"] == "complete"]
            is_primary = {(p["suite"], p["problem"], p["instance"]) for p in done} == inventory
            if is_primary:
                primary.add(model)
            coverage.append({"cohort_protocol": protocol, "model": model, "complete_pairs": len(done), "planned_pairs": len(inventory), "primary_model": is_primary})
        index = {}
        for call in report["calls"]:
            if call["model"] not in PAPER_MODELS:
                continue
            if call["invocation_id"] in global_ids:
                raise ValueError("Physical invocation appears in more than one audited report")
            global_ids.add(call["invocation_id"])
            extra = supplemental_claude(call, report["input_root"]) if read_supplemental else {}
            row = call | {"cohort_protocol": protocol, "input_root": report["input_root"], "primary_model": call["model"] in primary, **token_components(call, extra)}
            index[call["invocation_id"]] = row
            calls.append(row)
            if extra:
                supplements.append({key: row[key] for key in ("cohort_protocol", "model", "instance", "mode", "attempt", "stage", "classification", "invocation_id", "elapsed_seconds")} | extra)
        for pair in pairs:
            if pair["state"] != "complete":
                continue
            base = {key: pair[key] for key in ("model", "suite", "problem", "instance", "size", "seed")}
            base.update(cohort_protocol=protocol, input_root=report["input_root"], primary_model=pair["model"] in primary)
            for mode, policy in pair["policies"].items():
                if policy["audit_issues"] or policy["state"] != "complete":
                    raise ValueError("Complete pair contains unaudited policy")
                own = [index[call_id] for call_id in set(policy["call_ids"])]
                if any(c["model"] != pair["model"] or c["instance"] != pair["instance"] or c["classification"] not in ("shared_first", "policy") or not c.get("physical_call_confirmed", True) for c in own):
                    raise ValueError("Policy references an unrelated or unconfirmed call")
                for k in range(1, report["configuration"]["max_attempts"] + 1):
                    admitted = [c for c in own if c["attempt"] <= k and (c["stage"] != "reflection" or c["attempt"] < k)]
                    row = base | {"mode": mode, "budget": k, **policy["attempt_metrics"][str(k)]}
                    for name, stages in (("agent", AGENT_STAGES), ("generation", {"generation"}), ("reflection", {"reflection"}), ("judge", {"algorithm_verification"})):
                        selected = [c for c in admitted if c["stage"] in stages]
                        row[name + "_calls"] = len(selected)
                        for component in COMPONENTS:
                            values = [c[component] for c in selected]
                            field = name + "_" + component
                            row[field] = strict_sum(values)
                            row[field + "_missing_calls"] = sum(v is None for v in values)
                    budgets.append(row)
                final = budgets[-1].copy()
                first_q = next((b for b in budgets[-report["configuration"]["max_attempts"]:] if b["quality"]), None)
                final.update(first_quality_attempt=first_q["budget"] if first_q else None, tokens_to_quality=first_q["agent_accounted_tokens"] if first_q else None)
                policies.append(final)
    summary_groups = defaultdict(list)
    summary_measures = [f"{stage}_{component}" for stage in ("agent", "generation", "reflection", "judge") for component in COMPONENTS]
    for row in policies:
        for dimension, group in groups(row):
            summary_groups[(row["model"], row["mode"], dimension, group, row["primary_model"])].append(row)
    summaries, resource_curves = [], []
    for (model, mode, dimension, group, primary), records in summary_groups.items():
        base = {"model": model, "mode": mode, "dimension": dimension, "group": group, "primary_model": primary}
        summaries.extend(base | {"measure": measure} | statistics([r[measure] for r in records]) for measure in summary_measures)
        for limit in TOKEN_BUDGETS:
            successes = sum(r["quality"] and r["tokens_to_quality"] is not None and r["tokens_to_quality"] <= limit for r in records)
            unknown = sum(r["quality"] and r["tokens_to_quality"] is None for r in records)
            resource_curves.append(base | {"agent_token_budget": limit, "quality_successes": successes, "total_cases": len(records), "unlocated_quality_successes": unknown, "quality_rate": successes / len(records) if not unknown else None, "known_success_lower_bound": successes / len(records)})
    physical_groups = defaultdict(list)
    backend_fields = defaultdict(set)
    for call in calls:
        physical_groups[(call["cohort_protocol"], call["model"], call["backend"], call["stage"], call["classification"])].append(call)
        backend_fields[call["backend"]].update((call.get("usage") or {}).keys())
    raw_fields = []
    for (protocol, model, backend, stage, classification), records in physical_groups.items():
        physical = [r for r in records if r.get("physical_call_confirmed", True)]
        fields = sorted(backend_fields[backend])
        for field in fields:
            values = [integer((r.get("usage") or {}).get(field)) if r.get("usage_available") else None for r in physical]
            raw_fields.append({"cohort_protocol": protocol, "model": model, "backend": backend, "stage": stage, "classification": classification, "field": field, "recorded_requests": len(records), "physical_calls": len(physical), **statistics(values)})
    return {"coverage": coverage, "calls": calls, "supplemental_claude": supplements, "case_policies": policies, "budget_records": budgets, "group_summary": summaries, "physical_raw_fields": raw_fields, "resource_curves": resource_curves, "methods": {"accounted_tokens": "Codex: input_tokens + output_tokens. Claude: input_tokens + cache_creation_input_tokens + cache_read_input_tokens + output_tokens. Cache/reasoning components are not added twice. These are cumulative provider-token counts across CLI turns, not unique prompt lengths, monetary charges, or equal units of computation.", "gemini": "Raw provider fields retained. Full-context scalar is unavailable because AGY cache/input accounting is not explicitly established. Missing scalar costs are not zero.", "supplement": "Claude thinking is read from terminal usage.output_tokens_details.thinking_tokens, a subset of output. Numeric provider duration fields are saved separately; source event content is not copied.", "budget": "Scored generation/judge attempts <= k; reflection attempts < k. Shared first calls count in each policy counterfactual, once physically. Unsolved cases remain in resource-curve denominators; missing costs on solved cases make the rate unavailable, with known-success lower bound retained."}}
