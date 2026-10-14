"""Saved native/hybrid artifacts: paired outcomes, recovery costs, live reads and cohorts."""
from copy import deepcopy
import importlib
import json
from pathlib import Path
import shutil

import pytest

from src import campaign_report as reporter
from src import reporting_policy
from src.agent.paired_runner import digest, save


@pytest.fixture
def campaign(tmp_path):
    root = tmp_path / "campaign"
    source = Path(__file__).resolve().parents[1] / "src"
    shutil.copytree(source, root / "controller/src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    hashes = {str(p.relative_to(root / "controller")): digest(p) for p in (root / "controller").rglob("*") if p.is_file()}
    cases = []
    for i, size in enumerate(reporter.SIZES):
        case = {"suite": "bd", "problem": "ufl", "instance": f"ufl_{size}_seed000", "size": size, "seed": 0}
        directory = root / "dataset/bd/ufl" / case["instance"]
        save(directory / "data.json", {"m": 2, "n": 1, "f": [2, 3], "D": [[4], [1]]})
        save(directory / "meta.json", {"ground_truth_objective": 4})
        save(directory / "gold_monolithic.json", {"objective_value": 4})
        (directory / "prompt.txt").write_text("Synthetic UFL output fixture")
        case["sha256"] = {name: digest(directory / name) for name in ("data.json", "meta.json", "gold_monolithic.json", "prompt.txt")}
        cases.append(case)
    manifest = {"schema_version": 3, "backend": "native", "created_at": "2026-10-13T00:00:00+00:00", "instances": cases, "controller_sha256": hashes, "max_attempts": 3, "gap_threshold": 0.01, "judge_model": "gpt-6-astra", "solver_timeout": 60, "solver_threads": 1, "agent_timeout": 900, "workers": 2, "model_queue": [{"model": "gpt-6.1-sol", "backend": "codex-cli"}], "agent_execution": "docker", "solver_execution": "native-macos-seatbelt", "historical_reference_in_fresh_totals": False, "execution_environment": {"image_id": "sha256:fixture"}}
    manifest["protocol_id"] = reporter.identity(manifest)
    save(root / "campaign.json", manifest)
    return root, manifest


def set_backend(campaign, backend, model):
    root, manifest = campaign
    manifest["model_queue"] = [{"model": model, "backend": backend}]
    manifest.pop("protocol_id")
    manifest["protocol_id"] = reporter.identity(manifest)
    save(root / "campaign.json", manifest)


def invocation(backend, model, transport="docker", usage_available=True):
    result = {"backend": backend, "requested_model": model, "usage_available": usage_available}
    if transport == "docker":
        result.update(transport="docker", ephemeral=True, external_context=False, image_id="sha256:fixture", cleanup_verified=True)
    elif backend == "codex-cli":
        result.update(ephemeral=True, ignore_user_config=True, project_doc_max_bytes=0)
    else:
        result.update(fresh_conversation=True, resume=False, sandbox=True)
    return result


def write_call(directory, step, backend, model, call_id, *, ok=True, omitted=False, usage=None, committed=True, transport="docker", elapsed=2):
    call = directory / "calls" / call_id
    save(call / "request.json", {"step": step, "invocation_id": call_id, "request_sha256": "fixture"})
    target = {"agent": "solver.py", "llm_verifier": "alg_verifier.md", "reflection": "debug.md"}[step]
    contents = {"agent": "# synthetic solver; never executed by report\n", "llm_verifier": "VERDICT: PASS\nNOTES: synthetic review\n", "reflection": "Reflect on the error\n"}[step]
    if committed:
        (directory / target).write_text("" if omitted else contents)
    if usage is None:
        usage = {"input_tokens": 10, "output_tokens": 2}
    metrics = {"invocation_id": call_id, "ok": ok, "returncode": 0 if ok or omitted else 1, "elapsed_seconds": elapsed, "usage": usage, "terminal_state": "completed" if ok or omitted else "failed", "artifact_sha256": digest(directory / target) if committed else None}
    if omitted:
        metrics.update(file_delivery_failure=True, delivery_evidence_sha256={})
    inv = invocation(backend, model, transport, bool(usage))
    save(call / f"{step}_metrics.json", metrics)
    save(call / f"{step}_invocation.json", inv)
    if committed:
        save(directory / f"{step}_receipt.json", {"result": metrics, "request_sha256": "fixture"})
        save(directory / f"{step}_invocation.json", inv)
        save(directory / f"{step}_metrics.json", metrics)
    return metrics


def write_pair(campaign, case_index=0, *, plain_success=1, reflection_success=1, partial=False, omitted=False, backend_usage=None, transport="docker"):
    root, manifest = campaign
    spec = manifest["model_queue"][0]
    case = manifest["instances"][case_index]
    source = root / "dataset/bd/ufl" / case["instance"]
    shared_attempt = None
    for mode, solved_at in (("plain", plain_success), ("reflection", reflection_success)):
        directory = root / "runs" / spec["model"] / mode / "bd/ufl" / case["instance"]
        directory.mkdir(parents=True)
        for name in ("data.json", "prompt.txt"):
            shutil.copyfile(source / name, directory / name)
        save(directory / "identity.json", {"protocol_id": manifest["protocol_id"], "model": spec["model"], "mode": mode, "max_attempts": manifest["max_attempts"], "inputs": {n: case["sha256"][n] for n in ("data.json", "prompt.txt")}, "references": {n: case["sha256"][n] for n in ("meta.json", "gold_monolithic.json")}})
        first_dir = root / "runs" / spec["model"] / "plain/bd/ufl" / case["instance"] / "attempts/attempt_01"
        shared = {"id": reporter.identity({"protocol": manifest["protocol_id"], "model": spec["model"], "instance": case["instance"], "attempt": 1}), "canonical": str(first_dir), "generation_invocation_id": f"{case_index}-plain-1-agent", "judge_invocation_id": f"{case_index}-plain-1-judge"}
        save(directory / "shared_first.json", shared)
        attempts = []
        solved = None
        maximum = 1 if partial and mode == "reflection" else manifest["max_attempts"]
        for k in range(1, maximum + 1):
            if solved is not None:
                attempt = deepcopy(solved)
                attempt.update(attempt=k, status="skipped_previous_solved", reused_from_previous=True, first_success_attempt=solved["attempt"], metrics={})
                attempts.append(attempt)
                continue
            a = directory / "attempts" / f"attempt_{k:02d}"
            a.mkdir(parents=True)
            if mode == "reflection" and k == 1:
                for p in first_dir.iterdir():
                    if p.is_file():
                        shutil.copyfile(p, a / p.name)
                attempt = deepcopy(shared_attempt)
            else:
                generation = write_call(a, "agent", spec["backend"], spec["model"], f"{case_index}-{mode}-{k}-agent", usage=backend_usage, transport=transport)
                judge = write_call(a, "llm_verifier", "codex-cli", manifest["judge_model"], f"{case_index}-{mode}-{k}-judge", transport=transport)
                passed = solved_at is not None and k >= solved_at
                output = {"y": [0, 1] if passed else [1, 0], "x": [[0], [1]] if passed else [[1], [0]], "objective_value": 4 if passed else 6}
                save(a / "output.json", output)
                save(a / "execution.json", {"execution_seconds": 0.5, "solver_sha256": digest(a / "solver.py"), "output_sha256": digest(a / "output.json"), "exec_ok": True})
                attempt = {"attempt": k, "status": "ok" if passed else "feasible_not_optimal", "exec_ok": True, "verify_ok": True, "objective": output["objective_value"], "gold_objective": 4, "gap_to_gold": 0 if passed else 0.5, "llm_verdict": "pass", "reused_from_previous": False, "metrics": {"generation": generation, "algorithm_verification": judge, "execution_seconds": 0.5, "total_seconds": 4.5}}
                save(a / "score_receipt.json", {"result": attempt, "artifact_sha256": {n: digest(a / n) for n in ("solver.py", "output.json", "alg_verifier.md")}})
                save(a / "summary.json", attempt)
                if k == 1:
                    shared_attempt = deepcopy(attempt)
            if attempt["status"] == "ok":
                solved = deepcopy(attempt)
            elif mode == "reflection" and k < manifest["max_attempts"]:
                metrics = write_call(a, "reflection", spec["backend"], spec["model"], f"{case_index}-{mode}-{k}-reflection", ok=not omitted, omitted=omitted, transport=transport)
                attempt["metrics"]["reflection"] = metrics
            attempts.append(attempt)
        save(directory / "instance_summary.json", {"model": spec["model"], "backend": spec["backend"], "suite": "bd", "problem": "ufl", "instance": case["instance"], "reflection": mode == "reflection", "max_attempts": manifest["max_attempts"], "gap_threshold": manifest["gap_threshold"], "protocol_id": manifest["protocol_id"], "complete": maximum == manifest["max_attempts"], "attempts": attempts})
    return root / "runs" / spec["model"]


def assert_clean(report):
    assert not report["audit_errors"], report["audit_errors"]
    assert not report["snapshot"]["read_errors"], report["snapshot"]["read_errors"]
    assert not [p["audit_issues"] for pair in report["pairs"] for p in pair["policies"].values() if p["audit_issues"]]


@pytest.mark.parametrize("backend,model,usage", [("codex-cli", "gpt-6.1-sol", {"input_tokens": 10, "output_tokens": 2, "cached_input_tokens": 5}), ("claude-code-cli", "claude-sonnet-5-5", {"input_tokens": 6, "output_tokens": 7, "cache_read_input_tokens": 9}), ("agy-cli", "gemini-3.8-flash-medium", {"input_tokens": 10, "output_tokens": 5, "thinking_tokens": 2, "total_tokens": 15})])
@pytest.mark.parametrize("transport", ["docker", "native"])
def test_all_backends_shared_success_deduplicates_physical_calls(campaign, backend, model, usage, transport):
    set_backend(campaign, backend, model)
    write_pair(campaign, backend_usage=usage, transport=transport)
    report = reporter.build_campaign_report(campaign[0])
    assert_clean(report)
    assert report["coverage"]["completed_pairs"] == 1
    assert report["costs"]["all_recorded_physical"]["physical_cli_calls"] == 2
    assert report["costs"]["all_recorded_physical"]["solver_executions"] == 1
    pair = report["pairs"][0]
    assert pair["costs"]["plain_including_shared"]["physical_cli_calls"] == 2
    assert pair["costs"]["reflection_including_shared"]["physical_cli_calls"] == 2
    assert report["calls"][0]["usage"] == usage
    assert all(row["reflection_minus_plain"] == 0 for row in report["paired_curves"])
    assert all(row["paired_bootstrap_95"] is None for row in report["paired_curves"])


def test_exhaustion_reflection_omission_and_no_final_reflection(campaign):
    write_pair(campaign, plain_success=None, reflection_success=2, omitted=True)
    report = reporter.build_campaign_report(campaign[0])
    assert_clean(report)
    pair = report["pairs"][0]
    assert pair["state"] == "complete"
    assert pair["policies"]["plain"]["actual_attempts"] == 3
    assert pair["policies"]["reflection"]["actual_attempts"] == 2
    assert pair["policies"]["reflection"]["reused_slots"] == 1
    assert len(report["calls"]) == 9  # 4 unique solver generations, 4 judges, 1 reflection
    assert sum(c["stage"] == "reflection" for c in report["calls"]) == 1
    assert next(c for c in report["calls"] if c["stage"] == "reflection")["reflection_delivery_failure"]
    point = next(c for c in report["paired_curves"] if c["dimension"] == "overall" and c["budget"] == 2 and c["metric"] == "quality")
    assert point["reflection_only"] == 1 and point["reflection_minus_plain"] == 1


def test_retry_missing_usage_and_recovery_are_separate(campaign):
    run = write_pair(campaign)
    a = run / "plain/bd/ufl/ufl_small_seed000/attempts/attempt_01"
    write_call(a, "agent", "codex-cli", "gpt-6.1-sol", "failed-extra", ok=False, usage={}, committed=False, elapsed=3)
    write_call(a, "llm_verifier", "codex-cli", "gpt-6-astra", "malformed-judge", committed=False)
    save(a / "calls/malformed-judge/rejected.json", {"reason": "malformed verdict"})
    save(a / "calls/interrupted/request.json", {"step": "agent", "invocation_id": "interrupted"})
    report = reporter.build_campaign_report(campaign[0])
    assert_clean(report)
    assert len(report["calls"]) == 5
    classes = report["costs"]["by_classification"]
    assert classes["infrastructure_or_rejected"]["physical_cli_calls"] == 2
    assert classes["request_start_unconfirmed"]["unconfirmed_requests"] == 1
    generation = next(s for s in report["costs"]["all_recorded_physical"]["stages"] if s["stage"] == "generation")
    assert generation["usage_missing_calls"] == 1
    assert generation["raw_usage_fields"]["input_tokens"] == {"observed_sum": 10, "available_calls": 1, "missing_calls": 1, "complete": False}
    assert report["pairs"][0]["costs"]["plain_including_shared"]["physical_cli_calls"] == 2


def test_live_partial_never_enters_paired_failure_denominator(campaign):
    write_pair(campaign, plain_success=None, reflection_success=None, partial=True)
    report = reporter.build_campaign_report(campaign[0])
    assert_clean(report)
    assert report["coverage"]["completed_pairs"] == 0
    assert report["paired_curves"] == []
    assert report["pairs"][0]["state"] == "partial"
    assert report["coverage"]["sizes"] == [{"size": s, "inventory_cases": 1, "expected_pairs": 1, "completed_pairs": 0} for s in reporter.SIZES]


def test_tampered_saved_output_and_summary_are_not_audited(campaign):
    run = write_pair(campaign)
    save(run / "plain/bd/ufl/ufl_small_seed000/attempts/attempt_01/output.json", {"objective_value": 4, "y": [0, 0], "x": [[0], [0]]})
    report = reporter.build_campaign_report(campaign[0])
    assert report["coverage"]["completed_pairs"] == 0
    assert report["pairs"][0]["state"] == "audit_failed"
    issues = report["pairs"][0]["policies"]["plain"]["audit_issues"]
    assert any("changed scored artifact" in issue for issue in issues)
    assert any("independent output verification" in issue for issue in issues)


def test_changed_physical_usage_conflicts_with_receipt(campaign):
    run = write_pair(campaign)
    path = run / "plain/bd/ufl/ufl_small_seed000/attempts/attempt_01/calls/0-plain-1-agent/agent_metrics.json"
    data = json.loads(path.read_text())
    data["usage"]["input_tokens"] = 999
    save(path, data)
    report = reporter.build_campaign_report(campaign[0])
    assert report["coverage"]["completed_pairs"] == 0
    assert any("Physical metrics differ" in e["issue"] for e in report["audit_errors"])


def test_frozen_inventory_size_overrides_name_and_counts_reconcile(campaign):
    root, manifest = campaign
    write_pair(campaign, 0)
    write_pair(campaign, 1)
    write_pair(campaign, 2)
    report = reporter.build_campaign_report(root)
    assert_clean(report)
    assert report["coverage"]["complete"]
    assert sum(s["completed_pairs"] for s in report["coverage"]["sizes"]) == 3
    analysis = importlib.import_module("src.03_analysis")
    path = root / "runs/gpt-6.1-sol/plain/bd/ufl/ufl_small_seed000"
    assert analysis.infer_difficulty(path) == "small"
    instances = [{"difficulty": size, "solveE_at_i": [1], "solveV_at_i": [1], "solveQ_at_i": [1], "quality_at_i": [0]} for size in reporter.SIZES]
    assert sum(g["num_instances"] for g in analysis.aggregate_by_difficulty(instances).values() if g) == 3


def test_paired_bootstrap_reproducible_and_preserves_shared_first(campaign):
    write_pair(campaign, 0, plain_success=None, reflection_success=2)
    write_pair(campaign, 1, plain_success=2, reflection_success=None)
    report = reporter.build_campaign_report(campaign[0])
    assert_clean(report)
    assert report["paired_curves"] == reporter.paired_statistics(report["pairs"], 3)
    point = next(r for r in report["paired_curves"] if r["dimension"] == "overall" and r["budget"] == 2 and r["metric"] == "quality")
    assert point["paired_cases"] == 2 and point["paired_bootstrap_95"] == [-1.0, 1.0]
    first = next(r for r in report["paired_curves"] if r["dimension"] == "overall" and r["budget"] == 1 and r["metric"] == "quality")
    assert first["paired_bootstrap_95"] == [0.0, 0.0] and first["interval_degenerate"]


def test_reading_never_changes_source_or_writes_inside_campaign(campaign, tmp_path):
    write_pair(campaign)
    root = campaign[0]
    before = {str(p.relative_to(root)): digest(p) for p in root.rglob("*") if p.is_file()}
    report = reporter.build_campaign_report(root)
    reporter.export_campaign_report(report, tmp_path / "derived")
    assert (tmp_path / "derived/physical_calls.csv").is_file()
    assert "PROVISIONAL" in (tmp_path / "derived/report.md").read_text()
    with pytest.raises(ValueError, match="outside"):
        reporter.export_campaign_report(report, root / "report")
    assert before == {str(p.relative_to(root)): digest(p) for p in root.rglob("*") if p.is_file()}


def test_historical_sol_stays_excluded_and_fresh_requires_registered_identity(campaign, monkeypatch):
    root, manifest = campaign
    set_backend(campaign, "codex-cli", "gpt-5.6-sol")
    write_pair(campaign)
    excluded = reporter.build_campaign_report(root)
    assert excluded["coverage"]["expected_pairs"] == 0
    assert not excluded["calls"] and not excluded["paired_curves"]
    manifest["experiment_plan"] = {"name": reporting_policy.AUTHORIZED_FRESH_PLAN, "dataset_inventory_sha256": reporting_policy.AUTHORIZED_INVENTORY_SHA256, "reuse_generated_context": False, "reporting": {"mix_legacy_results_into_fresh_totals": False}}
    manifest.pop("protocol_id")
    manifest["protocol_id"] = reporter.identity(manifest)
    assert reporting_policy.model_exclusion("gpt-5.6-sol", manifest)
    monkeypatch.setattr(reporting_policy, "AUTHORIZED_FRESH_PROTOCOL_IDS", {manifest["protocol_id"]})
    assert reporting_policy.model_exclusion("gpt-5.6-sol", manifest) is None
    assert reporting_policy.model_exclusion("gpt-5.6-sol") is not None
    save(root / "campaign.json", manifest)
    assert reporting_policy.artifact_exclusion(root / "runs/gpt-5.6-sol/plain") is None


def test_snapshot_detects_replacement_without_locking(campaign):
    root, _ = campaign
    snapshot = reporter.Snapshot(root)
    path = root / "status.json"
    save(path, {"state": "running"})
    assert snapshot.json(path)["state"] == "running"
    save(path, {"state": "paused"})
    assert snapshot.json(path)["state"] == "running"
    assert snapshot.finish()["changed_during_read"] == ["status.json"]


def test_malformed_committed_physical_metrics_excludes_only_affected_pair(campaign):
    run = write_pair(campaign)
    write_pair(campaign, 1)
    path = run / "plain/bd/ufl/ufl_small_seed000/attempts/attempt_01/calls/0-plain-1-agent/agent_metrics.json"
    path.write_text('{"usage":')
    report = reporter.build_campaign_report(campaign[0])
    assert report["coverage"]["completed_pairs"] == 1
    assert report["pairs"][0]["state"] == "audit_failed"
    assert report["pairs"][1]["state"] == "complete"
    assert report["coverage"]["completed_instance_runs"] == 2
    assert report["snapshot"]["read_errors"]
    assert all(p["paired_cases"] == 1 for p in report["paired_curves"])


def test_shared_first_cannot_claim_a_new_invocation(campaign):
    run = write_pair(campaign)
    directory = run / "reflection/bd/ufl/ufl_small_seed000"
    a = directory / "attempts/attempt_01"
    for name in ("score_receipt.json", "summary.json", "agent_receipt.json"):
        path = a / name
        data = json.loads(path.read_text())
        result = data.get("result", data)
        metrics = result if name == "agent_receipt.json" else result["metrics"]["generation"]
        metrics["invocation_id"] = "pretend-shared-call"
        save(path, data)
    path = directory / "instance_summary.json"
    data = json.loads(path.read_text())
    data["attempts"][0]["metrics"]["generation"]["invocation_id"] = "pretend-shared-call"
    save(path, data)
    report = reporter.build_campaign_report(campaign[0])
    assert report["coverage"]["completed_pairs"] == 0
    assert report["coverage"]["completed_instance_runs"] == 0
    assert any("Shared-first scored outcomes" in i for i in report["pairs"][0]["policies"]["reflection"]["audit_issues"])


def test_successful_prefix_is_not_a_closed_paired_policy(campaign):
    run = write_pair(campaign)
    path = run / "plain/bd/ufl/ufl_small_seed000/instance_summary.json"
    data = json.loads(path.read_text())
    data["attempts"] = data["attempts"][:1]
    save(path, data)
    report = reporter.build_campaign_report(campaign[0])
    assert report["coverage"]["completed_pairs"] == 0
    assert report["pairs"][0]["state"] == "audit_failed"
    assert any("Claimed completion" in i for i in report["pairs"][0]["policies"]["plain"]["audit_issues"])


def test_request_before_call_gate_has_no_proven_physical_consumption(campaign):
    root, _ = campaign
    call = root / "runs/gpt-6.1-sol/plain/bd/ufl/ufl_small_seed000/attempts/attempt_01/calls/queued"
    save(call / "request.json", {"step": "agent", "invocation_id": "queued"})
    report = reporter.build_campaign_report(root)
    assert report["costs"]["all_recorded_physical"]["physical_cli_calls"] == 0
    assert report["costs"]["all_recorded_physical"]["unconfirmed_requests"] == 1
    assert report["costs"]["all_recorded_physical"]["stages"] == []
    assert not report["calls"][0]["physical_call_confirmed"]
    assert report["calls"][0]["usage"] is None
    assert report["coverage"]["completed_pairs"] == 0


def test_terminal_failure_uses_validated_actual_prefix_without_padding(campaign):
    from src.agent import policy_completion as completion
    root, manifest = campaign
    set_backend(campaign, "agy-cli", "gemini-3.1-pro-high")
    run = write_pair(campaign, plain_success=None, reflection_success=None, partial=True)
    directory = run / "reflection/bd/ufl/ufl_small_seed000"
    a = directory / "attempts/attempt_01"
    raw = json.loads((directory / "instance_summary.json").read_text())
    raw["attempts"][0]["metrics"].pop("reflection")
    save(directory / "instance_summary.json", raw)
    (a / "reflection_receipt.json").unlink()
    (a / "debug.md").write_text("")
    call = a / "calls/0-reflection-1-reflection"
    metrics = json.loads((call / "reflection_metrics.json").read_text())
    metrics.update(ok=False, returncode=0, terminal_state="completed", artifact_sha256=None, error="AGY did not complete an edit (exit 0); see saved evidence")
    save(call / "reflection_metrics.json", metrics)
    request = {"protocol": manifest["protocol_id"], "step": "reflection", "target": "debug.md", "prompt": "Synthetic reflection", "context": []}
    request["request_sha256"] = completion.identity(request)
    save(call / "request.json", request)
    receipt = completion.build_terminal_receipt(directory, root, user_decision="Fixture user closes this saved failure", recorded_at="2026-10-13T00:00:00Z", source_phase_id="fixture")
    receipt_path = directory / completion.TERMINAL_RECEIPT
    save(receipt_path, receipt)
    phase = {"base_manifest_sha256": digest(root / "campaign.json"), "changes": {"terminal_policy_completion": completion.TERMINAL_AMENDMENT}, "terminal_policy_receipts": {str(receipt_path.relative_to(root)): digest(receipt_path)}}
    phase["phase_id"] = completion.identity(phase)
    phase_path = root / "execution_phases/fixture/phase.json"
    save(phase_path, phase)
    save(root / "active_execution_phase.json", {"phase_id": phase["phase_id"], "phase_file": str(phase_path)})
    report = reporter.build_campaign_report(root)
    assert_clean(report)
    policy = report["pairs"][0]["policies"]["reflection"]
    assert policy["state"] == "complete"
    assert policy["actual_attempts"] == 1 and policy["unexecuted_attempt_slots"] == 2 and policy["reused_slots"] == 0
    assert report["coverage"]["completed_pairs"] == 1
    assert not policy["attempt_metrics"]["3"]["quality"]
    assert not json.loads((directory / "instance_summary.json").read_text())["complete"]


def test_paired_stage_costs_cancel_shared_work_and_keep_missing_differences_null(campaign):
    write_pair(campaign, plain_success=None, reflection_success=2)
    report = reporter.build_campaign_report(campaign[0])
    rows = report["paired_cost_differences"]
    generation = next(r for r in rows if r["stage"] == "generation" and r["measure"] == "input_tokens")
    assert (generation["plain"], generation["reflection"], generation["reflection_minus_plain"]) == (30, 20, -10)
    reflection = next(r for r in rows if r["stage"] == "reflection" and r["measure"] == "elapsed_seconds")
    assert (reflection["plain"], reflection["reflection"], reflection["reflection_minus_plain"]) == (0, 2, 2)
    pair = deepcopy(report["pairs"][0])
    pair["costs"]["plain_including_shared"]["stages"][1]["raw_usage_fields"]["input_tokens"]["complete"] = False
    assert any(r["reflection_minus_plain"] is None for r in reporter.paired_cost_differences([pair]))
