"""Active-work time accounting from audited campaign reports.

This module reads saved report data only. Durations are client-call and solver work sums; neither concurrent experiment wall time nor provider compute time is reconstructed. Scored outcomes are consumed unchanged from the audited reports.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics

from src.results.policy import PAPER_MODELS, run_identity


CASE_FIELDS = ("suite", "problem", "instance", "size", "seed")
MODES = ("plain", "reflection")
COMPONENT_STAGES = {"generation": "generation", "reflection": "reflection", "judge": "algorithm_verification"}
TIME_MEASURES = ("generation_seconds", "reflection_seconds", "agent_seconds", "judge_seconds", "solver_seconds", "active_work_seconds")
QUALITY_TIME_MEASURES = ("time_to_quality_agent_seconds", "time_to_quality_active_work_seconds")
DEFAULT_AGENT_BUDGETS = (0, 15, 30, 45, 60, 90, 120, 180, 240, 300, 450, 600, 900, 1200, 1800, 3600, 7200)
COMMITTED_CLASSIFICATIONS = {"shared_first", "policy"}
SCIENTIFIC_FIELDS = ("prompt_protocol", "prompt_policies", "algorithm_judge_prompt_sha256", "execution_environment", "workers", "max_attempts", "solver_timeout", "solver_threads", "agent_timeout", "solver_cooldown_seconds", "judge_model", "judge_effort", "gap_threshold", "shared_first_attempt", "stop_after_success", "reflection_policy", "resource_guard", "agent_execution", "solver_execution", "wrong_target_reflection_failure")


def _finite_duration(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def _duration(values):
    values = list(values)
    known = [float(value) for value in values if _finite_duration(value)]
    missing = len(values) - len(known)
    observed = math.fsum(known)
    return {"seconds": None if missing else observed, "observed_seconds": observed, "missing_durations": missing, "events": len(values)}


def _timing(calls, solvers):
    result = {}
    components = {name: [call.get("elapsed_seconds") for call in calls if call["stage"] == stage] for name, stage in COMPONENT_STAGES.items()}
    components["solver"] = [solver.get("execution_seconds") for solver in solvers]
    components["agent"] = components["generation"] + components["reflection"]
    components["active_work"] = components["agent"] + components["judge"] + components["solver"]
    for name, values in components.items():
        result.update({f"{name}_{key}": value for key, value in _duration(values).items()})
    return result


def _case_key(row):
    return tuple(row[field] for field in CASE_FIELDS)


def _deduplicate(rows, id_field, signature_fields):
    unique = {}
    for row in rows:
        identifier = row.get(id_field)
        if not identifier:
            raise ValueError(f"Missing {id_field}")
        if identifier in unique:
            if any(unique[identifier].get(field) != row.get(field) for field in signature_fields):
                raise ValueError(f"Conflicting duplicate {id_field}: {identifier}")
        else:
            unique[identifier] = row
    return unique


def _provenance(report):
    return {"cohort_protocol": report["configuration"]["protocol_id"], "input_root": report["input_root"]}


def _validate_reports(reports):
    if not reports:
        raise ValueError("At least one audited report is required")
    first = reports[0]["cohort_manifest"]
    expected_cases = {_case_key(case) for case in first["instances"]}
    if len(expected_cases) != len(first["instances"]):
        raise ValueError("Duplicate cases in manifest inventory")
    budget = reports[0]["configuration"]["max_attempts"]
    models_seen = set()
    for report in reports:
        manifest = report["cohort_manifest"]
        if not run_identity(manifest)["paper_run"]:
            raise ValueError("Timing comparison requires an authorized fresh cohort")
        if report.get("audit_errors") or report.get("snapshot", {}).get("read_errors") or report.get("snapshot", {}).get("changed_during_read"):
            raise ValueError("Timing comparison requires a clean, stable audited snapshot")
        if report["configuration"]["protocol_id"] != manifest["protocol_id"]:
            raise ValueError("Report and manifest protocol identities differ")
        if report["configuration"]["max_attempts"] != budget or manifest["max_attempts"] != budget or manifest["instances"] != first["instances"]:
            raise ValueError("Reports must use the same attempt budget and exact case inventory")
        for field in SCIENTIFIC_FIELDS:
            if manifest.get(field) != first.get(field):
                raise ValueError(f"Scientific settings differ: {field}")
        for field in ("reference_policy", "runtime_manifest_sha256"):
            if manifest.get("experiment_plan", {}).get(field) != first.get("experiment_plan", {}).get(field):
                raise ValueError(f"Reference/runtime settings differ: {field}")
        # File names of the older src/agent layout that the paper campaigns froze in their controllers.
        for name in ("src/agent/paired_runner.py", "src/agent/prompt_builder.py", "src/agent/output_verifier.py", "src/agent/native_solver.py"):
            if manifest.get("controller_sha256", {}).get(name) != first.get("controller_sha256", {}).get(name):
                raise ValueError(f"Scoring or prompt-construction source differs: {name}")
        pairs_by_model = defaultdict(list)
        for pair in report["pairs"]:
            if pair["model"] in PAPER_MODELS:
                pairs_by_model[pair["model"]].append(pair)
        for model, pairs in pairs_by_model.items():
            if model in models_seen:
                raise ValueError(f"Model appears in multiple reports: {model}")
            models_seen.add(model)
            keys = [_case_key(pair) for pair in pairs]
            if len(keys) != len(set(keys)) or set(keys) != expected_cases:
                raise ValueError(f"Model case inventory differs or is duplicated: {model}")
    return expected_cases, budget


def _validate_policy(pair, mode, budget):
    policy = pair["policies"][mode]
    if policy["state"] != "complete" or policy.get("audit_issues"):
        raise ValueError("Complete pair contains an incomplete or unaudited policy")
    actual = policy["actual_attempts"]
    if type(actual) is not int or not 0 < actual <= budget:
        raise ValueError("Invalid actual attempt count")
    previous = {name: False for name in ("execution", "verification", "quality")}
    for k in range(1, budget + 1):
        current = policy["attempt_metrics"].get(str(k))
        if not current or any(type(current.get(name)) is not bool for name in previous):
            raise ValueError("Missing or invalid audited cumulative outcomes")
        if not current["quality"] <= current["verification"] <= current["execution"] or any(previous[name] > current[name] for name in previous):
            raise ValueError("Audited outcomes violate monotonic E/V/Q nesting")
        previous = current


def _policy_events(pair, mode, calls_by_id, solvers_by_id):
    policy = pair["policies"][mode]
    calls, solvers = [], []
    for identifier in dict.fromkeys(policy["call_ids"]):
        if identifier not in calls_by_id:
            raise ValueError(f"Committed policy call is absent: {identifier}")
        call = calls_by_id[identifier]
        if (call["model"], call["instance"]) != (pair["model"], pair["instance"]):
            raise ValueError("Policy references a call from a different model/case")
        if call["classification"] not in COMMITTED_CLASSIFICATIONS or call["mode"] not in ("shared", mode) or call["stage"] not in COMPONENT_STAGES.values():
            raise ValueError("Committed policy references an additional or mismatched call")
        if not call.get("physical_call_confirmed"):
            raise ValueError("Committed policy contains a start-unconfirmed call")
        if type(call["attempt"]) is not int or not 1 <= call["attempt"] <= policy["actual_attempts"]:
            raise ValueError("Policy call is outside actual scored attempts")
        shared = call["attempt"] == 1 and call["stage"] != "reflection"
        if (call["mode"] == "shared") != shared or (call["classification"] == "shared_first") != shared:
            raise ValueError("First-attempt sharing attribution is inconsistent")
        if call["stage"] == "reflection" and mode != "reflection":
            raise ValueError("Plain policy contains a reflection call")
        calls.append(call)
    if sorted(call["attempt"] for call in calls if call["stage"] == "generation") != list(range(1, policy["actual_attempts"] + 1)):
        raise ValueError("Committed generation calls do not match actual scored attempts")
    for identifier in dict.fromkeys(policy["solver_ids"]):
        if identifier not in solvers_by_id:
            raise ValueError(f"Policy solver execution is absent: {identifier}")
        solver = solvers_by_id[identifier]
        if (solver["model"], solver["instance"]) != (pair["model"], pair["instance"]) or solver["mode"] not in ("shared", mode):
            raise ValueError("Policy references a solver from a different model/case/mode")
        if (solver["attempt"] == 1) != (solver["mode"] == "shared"):
            raise ValueError("First solver execution must have shared attribution")
        # An execution observed before its score commits remains physical work and never extends a policy.
        if 1 <= solver["attempt"] <= policy["actual_attempts"]:
            solvers.append(solver)
    return calls, solvers


def _dimensions(row):
    return (("overall", "all"), ("suite", row["suite"]), ("problem", f"{row['suite']}/{row['problem']}"), ("size", row["size"]))


def _percentile(values, probability):
    values = sorted(values)
    if not values:
        return None
    index = (len(values) - 1) * probability
    lower = math.floor(index)
    upper = math.ceil(index)
    return values[lower] + (values[upper] - values[lower]) * (index - lower)


def _statistics(values):
    values = [value for value in values if value is not None]
    return {"mean": statistics.fmean(values) if values else None, "median": statistics.median(values) if values else None, "p90": _percentile(values, 0.9), "p95": _percentile(values, 0.95), "max": max(values) if values else None, "min": min(values) if values else None}


def _summary_rows(case_rows):
    groups = defaultdict(list)
    for row in case_rows:
        for dimension, group in _dimensions(row):
            groups[(row["cohort_protocol"], row["model"], row["mode"], dimension, group)].append(row)
    result = []
    for (_, model, mode, dimension, group), rows in groups.items():
        for measure in (*TIME_MEASURES, *QUALITY_TIME_MEASURES):
            eligible = [row for row in rows if row["quality"]] if measure in QUALITY_TIME_MEASURES else rows
            values = [row[measure] for row in eligible]
            known = [value for value in values if value is not None]
            missing = len(values) - len(known)
            total = math.fsum(known) if not missing else None
            observed_name = measure.removesuffix("_seconds") + "_observed_seconds"
            observed_total = math.fsum(row[observed_name] for row in eligible)
            active_values = [row["active_work_seconds"] for row in rows]
            active_total = math.fsum(active_values) if all(value is not None for value in active_values) else None
            share = total / active_total if measure in TIME_MEASURES and total is not None and active_total else None
            result.append({"cohort_protocol": rows[0]["cohort_protocol"], "input_root": rows[0]["input_root"], "model": model, "mode": mode, "primary_model": rows[0]["primary_model"], "dimension": dimension, "group": group, "measure": measure, "population": "quality_successes" if measure in QUALITY_TIME_MEASURES else "all_completed_pairs", "cohort_cases": len(rows), "cases": len(eligible), "available_cases": len(known), "missing_cases": missing, "quality_successes": sum(row["quality"] for row in rows), "quality_rate": sum(row["quality"] for row in rows) / len(rows), **_statistics(values), "total": total, "observed_total": observed_total, "stage_share_of_active_work": share})
    return result


def _paired_rows(case_rows):
    pairs = defaultdict(dict)
    for row in case_rows:
        pairs[(row["cohort_protocol"], row["model"], row["instance"])][row["mode"]] = row
    differences = []
    for policies in pairs.values():
        plain, reflection = policies["plain"], policies["reflection"]
        status = "both" if plain["quality"] and reflection["quality"] else "plain_only" if plain["quality"] else "reflection_only" if reflection["quality"] else "neither"
        for measure in (*TIME_MEASURES, *QUALITY_TIME_MEASURES):
            left, right = plain[measure], reflection[measure]
            differences.append({**{key: plain[key] for key in ("cohort_protocol", "input_root", "model", *CASE_FIELDS, "primary_model")}, "measure": measure, "plain_quality": plain["quality"], "reflection_quality": reflection["quality"], "success_status": status, "plain": left, "reflection": right, "reflection_minus_plain": right - left if left is not None and right is not None else None})
    groups = defaultdict(list)
    for row in differences:
        for dimension, group in _dimensions(row):
            for status in ("all", row["success_status"]):
                groups[(row["cohort_protocol"], row["model"], dimension, group, row["measure"], status)].append(row)
    summaries = []
    for (_, model, dimension, group, measure, status), rows in groups.items():
        eligible = [row for row in rows if row["success_status"] == "both"] if measure in QUALITY_TIME_MEASURES else rows
        available = [row for row in eligible if row["reflection_minus_plain"] is not None]
        values = [row["reflection_minus_plain"] for row in available]
        counts = Counter(row["success_status"] for row in rows)
        summaries.append({"cohort_protocol": rows[0]["cohort_protocol"], "input_root": rows[0]["input_root"], "model": model, "primary_model": rows[0]["primary_model"], "dimension": dimension, "group": group, "measure": measure, "success_status": status, "population": "both_quality_successes" if measure in QUALITY_TIME_MEASURES else "all_completed_pairs", "cohort_pairs": len(rows), "paired_cases": len(eligible), "available_pairs": len(available), "missing_pairs": len(eligible) - len(available), "plain_mean": statistics.fmean(row["plain"] for row in available) if available else None, "reflection_mean": statistics.fmean(row["reflection"] for row in available) if available else None, **_statistics(values), "reflection_faster_pairs": sum(value < 0 for value in values), "reflection_slower_pairs": sum(value > 0 for value in values), "equal_time_pairs": sum(value == 0 for value in values), "both_successes": counts["both"], "plain_only_successes": counts["plain_only"], "reflection_only_successes": counts["reflection_only"], "neither_successes": counts["neither"]})
    return differences, summaries


def _physical_rows(calls, solvers, committed_solver_ids):
    events = []
    for call in calls:
        events.append({**call, "duration": call.get("elapsed_seconds"), "kind": "call"})
    for solver in solvers:
        classification = ("shared_first" if solver["mode"] == "shared" else "policy") if solver["solver_id"] in committed_solver_ids else "uncommitted_solver"
        events.append({**solver, "stage": "solver", "classification": classification, "duration": solver.get("execution_seconds"), "kind": "solver", "physical_call_confirmed": True})
    groups = defaultdict(list)
    for event in events:
        attribution = "committed_policy_path" if event["classification"] in COMMITTED_CLASSIFICATIONS else "additional"
        for category in ("all_recorded", attribution, event["classification"]):
            for stage in ("all", event["stage"]):
                for model in (event["model"], "__all__"):
                    groups[(event["cohort_protocol"], model, category, stage)].append(event)
    result = []
    for (protocol, model, category, stage), rows in groups.items():
        confirmed = [row for row in rows if row["physical_call_confirmed"]]
        costs = _duration(row["duration"] for row in confirmed)
        result.append({"cohort_protocol": protocol, "input_root": rows[0]["input_root"], "model": model, "category": category, "stage": stage, "recorded_requests": sum(row["kind"] == "call" for row in rows), "physical_calls": sum(row["kind"] == "call" for row in confirmed), "unconfirmed_requests": sum(row["kind"] == "call" and not row["physical_call_confirmed"] for row in rows), "solver_executions": sum(row["kind"] == "solver" for row in rows), "seconds": costs["seconds"], "observed_seconds": costs["observed_seconds"], "missing_durations": costs["missing_durations"], "counted_events": costs["events"]})
    return result


def _resource_curves(case_rows, agent_budgets):
    groups = defaultdict(list)
    for row in case_rows:
        for dimension, group in _dimensions(row):
            groups[(row["cohort_protocol"], row["model"], row["mode"], dimension, group)].append(row)
    curves = []
    for (_, model, mode, dimension, group), rows in groups.items():
        successful = [row for row in rows if row["quality"]]
        located = [row for row in successful if row["time_to_quality_agent_seconds"] is not None]
        unknown = len(successful) - len(located)
        for budget in agent_budgets:
            count = sum(row["time_to_quality_agent_seconds"] <= budget for row in located)
            curves.append({"cohort_protocol": rows[0]["cohort_protocol"], "input_root": rows[0]["input_root"], "model": model, "mode": mode, "primary_model": rows[0]["primary_model"], "dimension": dimension, "group": group, "agent_budget_seconds": budget, "quality_successes": count, "quality_rate": count / len(rows) if not unknown else None, "quality_rate_lower_bound": count / len(rows), "quality_rate_upper_bound": (count + unknown) / len(rows), "total_cases": len(rows), "final_quality_successes": len(successful), "unlocated_quality_successes": unknown})
    return curves


def _tail_rows(case_rows, count=5):
    groups = defaultdict(list)
    for row in case_rows:
        groups[(row["model"], row["mode"])].append(row)
    tails = []
    for rows in groups.values():
        for measure in ("agent_seconds", "active_work_seconds"):
            known = sorted((row for row in rows if row[measure] is not None), key=lambda row: (-row[measure], row["instance"]))
            for rank, row in enumerate(known[:count], 1):
                tails.append({**{key: row[key] for key in ("cohort_protocol", "input_root", "model", *CASE_FIELDS, "mode", "primary_model", "quality", "actual_attempts")}, "measure": measure, "rank": rank, "seconds": row[measure]})
    return tails


def _accounting_checks(case_rows, calls, solvers):
    physical = _timing(calls, solvers)
    shared = _timing([call for call in calls if call["mode"] == "shared"], [solver for solver in solvers if solver["mode"] == "shared"])
    checks = []
    for measure in TIME_MEASURES:
        observed = measure.removesuffix("_seconds") + "_observed_seconds"
        logical = _duration(row[measure] for row in case_rows)["seconds"]
        logical_observed = math.fsum(row[observed] for row in case_rows)
        expected = physical[measure] + shared[measure] if physical[measure] is not None and shared[measure] is not None else None
        residual = logical - expected if logical is not None and expected is not None else None
        observed_residual = logical_observed - physical[observed] - shared[observed]
        passed = (residual is None or math.isclose(residual, 0, abs_tol=1e-8)) and math.isclose(observed_residual, 0, abs_tol=1e-8)
        if not passed:
            raise ValueError(f"Shared-cost subtraction does not reconcile: {measure}")
        checks.append({"measure": measure, "scope": "all_completed_audited_pairs", "cohort_protocols": sorted({row["cohort_protocol"] for row in case_rows}), "input_roots": sorted({row["input_root"] for row in case_rows}), "logical_policy_seconds": logical, "physical_committed_seconds": physical[measure], "shared_once_seconds": shared[measure], "logical_minus_shared_minus_physical_seconds": residual, "logical_observed_seconds": logical_observed, "observed_residual_seconds": observed_residual, "passed": passed})
    return checks


def build_time_analysis(reports, *, agent_budgets=DEFAULT_AGENT_BUDGETS):
    """Return serializable policy, budget, distribution, and physical accounting.

    Each logical policy includes its shared first call and solver execution. A reflection recorded at attempt k enters budget k+1, while candidate generation, judge and solver work at k enter budget k. No early-success padding creates work. A missing required duration makes the corresponding total null.
    """
    expected_cases, budget = _validate_reports(reports)
    if not agent_budgets or any(not _finite_duration(value) for value in agent_budgets):
        raise ValueError("Agent-time budgets must be finite nonnegative numbers")
    agent_budgets = sorted(set(agent_budgets))
    all_calls, all_solvers, coverage = [], [], []
    sources = []
    for report in reports:
        provenance = _provenance(report)
        all_calls.extend(call | provenance for call in report["calls"] if call["model"] in PAPER_MODELS)
        all_solvers.extend(solver | provenance for solver in report["solver_executions"] if solver["model"] in PAPER_MODELS)
        by_model = defaultdict(list)
        for pair in report["pairs"]:
            if pair["model"] in PAPER_MODELS:
                by_model[pair["model"]].append(pair)
        for model, pairs in by_model.items():
            complete = [pair for pair in pairs if pair["state"] == "complete"]
            coverage.append({**provenance, "model": model, "planned_pairs": len(expected_cases), "completed_pairs": len(complete), "primary_model": {_case_key(pair) for pair in complete} == expected_cases, "pair_states": dict(Counter(pair["state"] for pair in pairs))})
        sources.append({**provenance, **report.get("_time_source", {}), "generated_at_utc": report.get("generated_at_utc"), "snapshot": report.get("snapshot")})
    calls_by_id = _deduplicate(all_calls, "invocation_id", ("cohort_protocol", "input_root", "model", "instance", "mode", "stage", "attempt", "classification", "physical_call_confirmed", "elapsed_seconds"))
    solvers_by_id = _deduplicate(all_solvers, "solver_id", ("cohort_protocol", "input_root", "model", "instance", "mode", "attempt", "execution_seconds"))
    primary = {row["model"] for row in coverage if row["primary_model"]}
    case_rows, budget_rows = [], []
    committed_solver_ids = set()
    completed_call_ids, completed_solver_ids = set(), set()
    for report in reports:
        provenance = _provenance(report)
        for pair in report["pairs"]:
            if pair["model"] not in PAPER_MODELS:
                continue
            for policy in pair["policies"].values():
                for identifier in policy["solver_ids"]:
                    solver = solvers_by_id.get(identifier)
                    if solver and 1 <= solver["attempt"] <= policy["actual_attempts"]:
                        committed_solver_ids.add(identifier)
            if pair["state"] != "complete":
                continue
            for mode in MODES:
                _validate_policy(pair, mode, budget)
            if pair["policies"]["plain"]["attempt_metrics"]["1"] != pair["policies"]["reflection"]["attempt_metrics"]["1"]:
                raise ValueError("Shared first-attempt outcomes differ")
            events = {mode: _policy_events(pair, mode, calls_by_id, solvers_by_id) for mode in MODES}
            shared_calls = [{call["invocation_id"] for call in events[mode][0] if call["mode"] == "shared"} for mode in MODES]
            shared_solvers = [{solver["solver_id"] for solver in events[mode][1] if solver["mode"] == "shared"} for mode in MODES]
            if shared_calls[0] != shared_calls[1] or shared_solvers[0] != shared_solvers[1]:
                raise ValueError("Logical policies reference different shared-first events")
            base = {**provenance, **{key: pair[key] for key in ("model", *CASE_FIELDS)}, "primary_model": pair["model"] in primary}
            for mode in MODES:
                policy = pair["policies"][mode]
                calls, solvers = events[mode]
                completed_call_ids.update(call["invocation_id"] for call in calls)
                completed_solver_ids.update(solver["solver_id"] for solver in solvers)
                first_quality = None
                first_quality_row = None
                for k in range(1, budget + 1):
                    cumulative_calls = [call for call in calls if call["attempt"] < k or (call["attempt"] == k and call["stage"] != "reflection")]
                    cumulative_solvers = [solver for solver in solvers if solver["attempt"] <= k]
                    row = {**base, "mode": mode, "budget": k, **policy["attempt_metrics"][str(k)], **_timing(cumulative_calls, cumulative_solvers)}
                    budget_rows.append(row)
                    if first_quality is None and row["quality"]:
                        first_quality, first_quality_row = k, row
                row = {**base, "mode": mode, "actual_attempts": policy["actual_attempts"], "reused_slots": policy.get("reused_slots", 0), "unexecuted_attempt_slots": policy.get("unexecuted_attempt_slots", 0), **policy["attempt_metrics"][str(budget)], "first_quality_attempt": first_quality, **_timing(calls, solvers)}
                for name in ("agent", "active_work"):
                    for suffix in ("seconds", "observed_seconds", "missing_durations"):
                        row[f"time_to_quality_{name}_{suffix}"] = first_quality_row[f"{name}_{suffix}"] if first_quality_row else None
                for name in ("generation", "reflection", "judge", "solver"):
                    row[f"{name}_share_active_work"] = row[f"{name}_seconds"] / row["active_work_seconds"] if row["active_work_seconds"] else None
                case_rows.append(row)
    differences, paired_summary = _paired_rows(case_rows)
    checks = _accounting_checks(case_rows, [calls_by_id[identifier] for identifier in completed_call_ids], [solvers_by_id[identifier] for identifier in completed_solver_ids])
    metadata = {
        "schema_version": "decompbench-active-time-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "sources": sources,
        "primary_models": sorted(primary),
        "common_inventory_cases": len(expected_cases),
        "max_attempts": budget,
        "agent_budgets_seconds": agent_budgets,
        "definitions": {
            "agent_seconds": "Sum of committed generation and reflection worker-client elapsed_seconds, including CLI/network/internal retry and wrapper credential-checkpoint/event-processing/version-check work; shared first generation belongs to each logical policy.",
            "judge_seconds": "Committed algorithm_verification worker-client elapsed_seconds; separate from the candidate agent.",
            "solver_seconds": "Unique recorded execution_seconds attributed to the logical policy, including shared first execution.",
            "active_work_seconds": "Agent plus judge plus solver work; cumulative activity, not wall-clock campaign latency, full container/queue time, or provider compute time.",
            "budget_boundary": "Generation, judge and solver at attempt <= k; reflection at attempt < k. A reflection at k prepares candidate k+1.",
            "missing_duration": "Strict total is null if any required duration is absent, invalid or nonfinite. Observed sums are lower bounds; no events means zero.",
            "early_stop": "Only committed physical call IDs and actual solver executions count. Reused scored slots contribute no new work.",
            "time_to_quality": "Cumulative work through the first audited Q success. Unsuccessful policies remain null and stay in fixed-budget success denominators.",
            "resource_curve": "Empirical replay of each recorded committed policy path at fixed cumulative candidate-agent budgets. No reruns or altered scores; fixed client-time budgets exclude judge and solver work.",
            "physical": "Invocation IDs and solver IDs count once. Committed, additional/rejected, and start-unconfirmed requests are separately tabulated across the full recorded eligible cohort, including incomplete cases. Unconfirmed requests are excluded from proven physical work.",
            "stage_shares": "Ratio of group stage duration sums to group active-work duration sums; null if any required group duration is missing.",
            "quantiles": "Linear interpolation at index (n-1)*p; distributions describe case variation in the saved run.",
            "partial_models": "Only models with complete audited coverage of the exact common inventory are primary. Other completed pairs are labeled partial and never pooled into primary model comparisons.",
            "provider_time": "Provider-reported durations, when available in supplemental raw-event evidence, are separate optional diagnostics and do not replace client elapsed time.",
            "qualifications": "Retain the three documented native SIGTERM recorded-score qualifications and two Gemini empty-SUCCESS events. This analysis does not rescore outcomes; GPT-5.6 Sol is excluded globally.",
        },
    }
    return {
        "metadata": metadata,
        "coverage": coverage,
        "case_policies": case_rows,
        "budget_records": budget_rows,
        "group_summary": _summary_rows(case_rows),
        "paired_differences": differences,
        "paired_summary": paired_summary,
        "physical_summary": _physical_rows(list(calls_by_id.values()), list(solvers_by_id.values()), committed_solver_ids),
        "resource_curves": _resource_curves(case_rows, agent_budgets),
        "tail_cases": _tail_rows(case_rows),
        "accounting_checks": checks,
    }


def write_analysis(analysis, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "time_analysis.json").write_text(json.dumps(analysis, indent=2, allow_nan=False) + "\n")
    (output_dir / "manifest.json").write_text(json.dumps(analysis["metadata"], indent=2, allow_nan=False) + "\n")
    for name, rows in analysis.items():
        if not isinstance(rows, list):
            continue
        with (output_dir / f"{name}.csv").open("w", newline="") as stream:
            if not rows:
                continue
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows({key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value for key, value in row.items()} for row in rows)

