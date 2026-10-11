"""User-adjudicated failure closes coverage without inventing model attempts."""
from copy import deepcopy
from datetime import datetime, timezone
import csv
import importlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.agent import policy_completion as completion
from src.agent.paired_runner import PairedRunner, ProtocolError, save
from src.dashboard_server import Dashboard, export_csv
from src.native_campaign import pair_complete
from src.native_execution_phase import USER_PAUSE_GUARD_CHANGE, ensure_execution_allowed, validate_phase, install_overrides


MODEL = "gemini-3.1-pro-high"


@pytest.fixture
def closed_policy(tmp_path):
    root = tmp_path / "campaign"
    case = {"suite": "bd", "problem": "ufl", "instance": "ufl_small_seed000", "size": "small", "seed": 0}
    source = root / "dataset/bd/ufl/ufl_small_seed000"
    source.mkdir(parents=True)
    (source / "data.json").write_text('{"m":2,"n":1,"f":[2,3],"D":[[4],[1]]}')
    (source / "prompt.txt").write_text("Implement Benders decomposition.")
    (source / "meta.json").write_text('{"ground_truth_objective":4}')
    (source / "gold_monolithic.json").write_text('{"objective_value":4}')
    case["sha256"] = {p.name: completion.digest(p) for p in source.iterdir()}
    manifest = {"backend": "native", "protocol_id": "base", "workers": 2, "max_attempts": 10,
                "gpt_order": [], "agy_order": [MODEL], "instances": [case]}
    save(root / "campaign.json", manifest)
    generation = {"ok": False, "returncode": 0, "terminal_state": "completed", "artifact_sha256": None,
                  "invocation_id": "generation-1", "usage": {"input_tokens": 17}}
    first = {"attempt": 1, "status": "agent_failed", "exec_ok": False, "verify_ok": False,
             "gold_objective": 4, "objective": None, "gap_to_gold": None, "metrics": {"generation": generation}}
    shared = {"id": completion.identity({"protocol": "base", "model": MODEL, "instance": case["instance"], "attempt": 1}),
              "generation_invocation_id": "generation-1", "judge_invocation_id": None}
    directories = {}
    for mode in ("plain", "reflection"):
        directory = root / "runs" / MODEL / mode / "bd/ufl" / case["instance"]
        directory.mkdir(parents=True)
        for name in ("data.json", "prompt.txt"):
            (directory / name).write_bytes((source / name).read_bytes())
        save(directory / "identity.json", {"protocol_id": "base", "model": MODEL, "mode": mode, "max_attempts": 10,
                                            "inputs": {n: case["sha256"][n] for n in ("data.json", "prompt.txt")},
                                            "references": {n: h for n, h in case["sha256"].items() if n not in ("data.json", "prompt.txt")}})
        save(directory / "shared_first.json", shared)
        attempts = [deepcopy(first)]
        if mode == "plain":
            attempts.append({"attempt": 2, "status": "ok", "exec_ok": True, "verify_ok": True,
                             "objective": 4, "gold_objective": 4, "gap_to_gold": 0,
                             "metrics": {"generation": {"invocation_id": "generation-2", "usage": {"input_tokens": 11}}}})
            attempts.extend({"attempt": k, "status": "skipped_previous_solved", "reused_from_previous": True,
                             "first_success_attempt": 2, "exec_ok": True, "verify_ok": True, "objective": 4,
                             "gap_to_gold": 0, "metrics": {}} for k in range(3, 11))
        save(directory / "instance_summary.json", {**{k: case[k] for k in ("instance", "suite", "problem")},
                                                    "model": MODEL, "backend": "agy-cli", "reflection": mode == "reflection",
                                                    "shared_first_attempt": True, "protocol_id": "base", "max_attempts": 10,
                                                    "gap_threshold": 0.01, "complete": mode == "plain", "attempts": attempts})
        directories[mode] = directory
    directory = directories["reflection"]
    attempt = directory / "attempts/attempt_01"
    attempt.mkdir(parents=True)
    (attempt / "solver.py").write_text("")
    (attempt / "debug.md").write_text("")
    save(attempt / "score_receipt.json", {"result": first, "artifact_sha256": {"solver.py": completion.digest(attempt / "solver.py")}})
    save(attempt / "summary.json", first)
    save(attempt / "agent_receipt.json", {"result": generation})
    save(attempt / "agent_invocation.json", {"fresh_conversation": True, "resume": False, "sandbox": True})
    request = {"protocol": "base", "step": "reflection", "target": "debug.md", "prompt": "Reflect on the saved failure.", "context": []}
    request["request_sha256"] = completion.identity(request)
    for index in range(3):
        invocation = f"reflection-{index}"
        call = attempt / "calls" / invocation
        call.mkdir(parents=True)
        save(call / "request.json", request | {"invocation_id": invocation})
        save(call / "reflection_metrics.json", {"ok": False, "returncode": 0, "terminal_state": "completed",
                                               "artifact_sha256": None, "invocation_id": invocation,
                                               "error": "AGY did not complete an edit (exit 0); see reflection_stderr.log"})
    before = {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    receipt = completion.build_terminal_receipt(directory, root, user_decision="End this saved reflection as failed.",
                                                recorded_at="2026-10-10T16:00:00+00:00", source_phase_id="parent")
    receipt_path = directory / completion.TERMINAL_RECEIPT
    save(receipt_path, receipt)
    phase_dir = root / "execution_phases/terminal"
    phase_dir.mkdir(parents=True)
    helper = phase_dir / "policy_completion.py"
    helper.write_bytes(Path(completion.__file__).read_bytes())
    wrapper = phase_dir / "native_execution_phase.py"
    wrapper.write_text("frozen wrapper fixture")
    save(phase_dir / "preserved_artifacts.json", {str(directory.relative_to(root) / "instance_summary.json"): completion.digest(directory / "instance_summary.json")})
    phase = {"base_protocol_id": "base", "base_manifest_sha256": completion.digest(root / "campaign.json"),
             "wrapper_sha256": completion.digest(wrapper), "policy_completion_sha256": completion.digest(helper),
             "preserved_inventory_sha256": completion.digest(phase_dir / "preserved_artifacts.json"),
             "changes": {"active_codex_invocation_limit": {"from": 1, "to": 2}, "terminal_policy_completion": completion.TERMINAL_AMENDMENT},
             "terminal_policy_receipts": {str(receipt_path.relative_to(root)): completion.digest(receipt_path)}}
    phase["phase_id"] = completion.identity(phase)
    phase_path = phase_dir / "phase.json"
    save(phase_path, phase)
    save(root / "active_execution_phase.json", {"phase_id": phase["phase_id"], "phase_file": str(phase_path)})
    save(root / "status.json", {"state": "paused", "active_instances": {}})
    save(root / "monitor/status.json", {"state": "paused", "sampled_at_utc": datetime.now(timezone.utc).isoformat(), "thermal": "No warning"})
    return SimpleNamespace(root=root, case=case, source=source, directory=directory, plain=directories["plain"],
                           attempt=attempt, receipt=receipt, receipt_path=receipt_path, before=before,
                           phase=phase, phase_path=phase_path, wrapper=wrapper)


def test_adjudication_preserves_raw_files_and_counts_one_actual_attempt(closed_policy):
    f = closed_policy
    summary = completion.load_policy_summary(f.directory)
    assert summary["complete"] and summary["outcome"] == "failed"
    assert len(summary["attempts"]) == summary["actual_attempts"] == 1
    assert summary["unexecuted_attempt_slots"] == 9 and summary["reused_slots"] == 0
    assert summary["attempts"][0]["status"] == "agent_failed"
    assert not summary["attempt_budget_exhausted"]
    assert pair_complete(f.root, MODEL, f.case, "base")
    for name, contents in f.before.items():
        assert (f.root / name).read_bytes() == contents


def test_direct_resume_never_calls_or_rewrites_closed_pair(closed_policy, monkeypatch):
    f = closed_policy
    class NoCalls:
        def protocol(self):
            return {"backend": "agy-cli", "model": MODEL}
        def edit_file(self, *args, **kwargs):
            pytest.fail("Closed policy attempted a model call")
    client = NoCalls()
    runner = PairedRunner(client=client, judge_client=client, reflection_client=client, protocol_id="base",
                          model=MODEL, suite="bd", problem=["ufl"], max_attempts=10)
    monkeypatch.setattr(runner, "prepare", lambda *args: pytest.fail("Closed pair was prepared again"))
    monkeypatch.setattr(runner, "run_attempt", lambda *args: pytest.fail("Closed pair was executed again"))
    summaries = runner.run_pair(f.source, f.root / "runs" / MODEL)
    assert summaries["plain"]["complete"] and summaries["reflection"]["complete"]
    assert (f.directory / "instance_summary.json").read_bytes() == f.before[str((f.directory / "instance_summary.json").relative_to(f.root))]


@pytest.mark.parametrize("changed", ["receipt", "missing_receipt", "score", "data", "identity", "request", "successful_call", "new_call", "reflection_receipt", "later_attempt"])
def test_inconsistent_completion_evidence_fails_closed(closed_policy, changed):
    f = closed_policy
    if changed == "missing_receipt":
        f.receipt_path.unlink()
    elif changed == "reflection_receipt":
        save(f.attempt / "reflection_receipt.json", {"result": {"ok": True}})
    elif changed == "later_attempt":
        (f.directory / "attempts/attempt_02").mkdir()
    elif changed == "new_call":
        new = f.attempt / "calls/new"
        new.mkdir()
        save(new / "reflection_metrics.json", {"ok": True})
    else:
        path = {"receipt": f.receipt_path, "score": f.attempt / "score_receipt.json", "data": f.directory / "data.json",
                "identity": f.directory / "identity.json", "request": f.attempt / "calls/reflection-0/request.json",
                "successful_call": f.attempt / "calls/reflection-0/reflection_metrics.json"}[changed]
        path.write_text('{}')
    with pytest.raises(completion.TerminalCompletionError):
        completion.load_policy_summary(f.directory)
    with pytest.raises(ProtocolError):
        pair_complete(f.root, MODEL, f.case, "base")


def test_different_case_is_not_implicitly_closed(closed_policy):
    f = closed_policy
    other = dict(f.case, instance="ufl_small_seed001")
    assert not pair_complete(f.root, MODEL, other, "base")


def test_dashboard_and_csv_separate_completion_from_first_attempt(closed_policy):
    f = closed_policy
    snapshot = Dashboard(f.root).snapshot()
    assert snapshot["state"] == "paused" and snapshot["totals"]["active"] == 0
    assert snapshot["totals"]["completed"] == snapshot["totals"]["planned"] == 1
    assert snapshot["totals"]["passed"] == 0
    assert snapshot["models"][0]["policy_finished"] == {"plain": 1, "reflection": 1}
    assert snapshot["models"][0]["policy_terminal_failures"] == {"plain": 0, "reflection": 1}
    assert snapshot["usage"]["generation"]["input_tokens"] == 28
    row = next(csv.DictReader(io.StringIO(export_csv(snapshot).decode())))
    assert row["status"] == "agent_failed" and row["plain_outcome"] == "passed"
    assert row["reflection_outcome"] == "terminal_failed"
    assert Dashboard(f.root).details(MODEL, f.case["instance"])["policies"]["reflection"]["unexecuted_attempt_slots"] == 9


def test_analysis_and_report_preserve_attempt_count_and_failed_budget_curve(closed_policy):
    f = closed_policy
    analysis = importlib.import_module("src.03_analysis")
    report = importlib.import_module("src.05_report")
    result = analysis.analyze_instance(f.directory, 0.01, False)
    assert result["actual_attempts"] == result["num_attempts"] == 1
    assert result["unexecuted_attempt_slots"] == 9
    assert result["solveE_at_i"] == result["solveV_at_i"] == result["solveQ_at_i"] == [0] * 10
    summary = completion.load_policy_summary(f.directory)
    assert report.audit_instance(("reflection", "bd", "ufl", f.case["instance"]), f.directory / "instance_summary.json",
                                 summary, f.source, {"max_attempts": 10, "gap_threshold": 0.01}) == []
    assert report.attempt_scores(summary["attempts"], 4, 0.01, 10)["quality"] is False


@pytest.mark.parametrize("changed", [None, "helper", "receipt", "summary"])
def test_phase_verifies_helper_receipt_and_protected_original(closed_policy, changed):
    f = closed_policy
    if changed:
        path = {"helper": f.phase_path.parent / "policy_completion.py", "receipt": f.receipt_path,
                "summary": f.directory / "instance_summary.json"}[changed]
        path.write_text("changed")
        with pytest.raises((ValueError, KeyError)):
            validate_phase(f.root, f.phase_path, executing_file=f.wrapper)
    else:
        assert validate_phase(f.root, f.phase_path, executing_file=f.wrapper)["phase_id"] == f.phase["phase_id"]


def test_frozen_campaign_override_honors_closed_pair(closed_policy, monkeypatch):
    f = closed_policy
    import sys
    old_module = sys.modules["src.agent.policy_completion"]
    monkeypatch.setitem(sys.modules, "src.agent.policy_completion", old_module)
    def legacy_complete(*args):
        pytest.fail("Adjudicated pair was sent to the legacy ten-entry predicate")
    campaign = SimpleNamespace(pair_complete=legacy_complete, save=lambda *args: None, supervise=lambda *args, **kwargs: None)
    paired = SimpleNamespace(ProtocolError=ProtocolError, runtime_lock=lambda name: None)
    class Client:
        def edit_file(self, *args, **kwargs):
            pytest.fail("No model call is needed to install completion support")
    class Agy(Client):
        pass
    state = f.root / "state"
    state.mkdir()
    install_overrides(f.root, f.phase_path, f.phase, campaign=campaign, paired=paired,
                      codex=SimpleNamespace(CodexClient=Client), agy=SimpleNamespace(AgyClient=Agy),
                      state_directory=state, wrapper_path=f.wrapper)
    assert campaign.pair_complete(f.root, MODEL, f.case, "base")


def test_user_pause_blocks_execution_but_allows_offline_verification(tmp_path):
    phase = {"changes": {"user_pause_guard": USER_PAUSE_GUARD_CHANGE}}
    save(tmp_path / "pause_request.json", {"resume_requires_explicit_user_request": True})
    ensure_execution_allowed(tmp_path, "verify", phase)
    for command in ("run", "watch"):
        with pytest.raises(ValueError, match="paused by the user"):
            ensure_execution_allowed(tmp_path, command, phase)
    save(tmp_path / "pause_request.json", {"resume_requires_explicit_user_request": False})
    ensure_execution_allowed(tmp_path, "run", phase)
