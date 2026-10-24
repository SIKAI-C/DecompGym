"""Token accounting avoids double-counting and preserves unknown resource costs."""
from copy import deepcopy
import json

from src.results.token_analysis import build_token_analysis, strict_sum, supplemental_claude, token_components


def test_codex_cache_and_reasoning_are_subsets():
    row = token_components({"backend": "codex-cli", "usage_available": True, "usage": {"input_tokens": 1000, "cached_input_tokens": 800, "cache_write_input_tokens": 0, "output_tokens": 200, "reasoning_output_tokens": 150}})
    assert row["accounted_tokens"] == 1200
    assert row["reasoning_tokens"] == 150
    assert row["cache_read_tokens"] == 800


def test_claude_input_categories_are_added_but_thinking_is_not():
    row = token_components({"backend": "claude-code-cli", "usage_available": True, "usage": {"input_tokens": 8, "cache_creation_input_tokens": 12046, "cache_read_input_tokens": 37799, "output_tokens": 3930}}, {"thinking_tokens": 649})
    assert row["context_input_tokens"] == 49853
    assert row["accounted_tokens"] == 53783
    assert row["reasoning_tokens"] == 649


def test_missing_is_unknown_and_absent_stage_is_zero():
    assert strict_sum([]) == 0
    assert strict_sum([1, None]) is None
    row = token_components({"backend": "claude-code-cli", "usage_available": True, "usage": {"input_tokens": 8, "output_tokens": 20}})
    assert row["accounted_tokens"] is None
    assert row["reasoning_tokens"] is None


def test_agy_cache_exceeding_input_is_kept_raw():
    row = token_components({"backend": "agy-cli", "usage_available": True, "usage": {"input_tokens": 278, "cache_read_tokens": 30214, "output_tokens": 512, "thinking_tokens": 200, "total_tokens": 790}})
    assert row["accounted_tokens"] is None
    assert row["provider_reported_total_tokens"] == 790
    assert row["cache_read_tokens"] == 30214


def synthetic_report():
    calls = []
    for call_id, mode, stage, attempt, input_tokens in [("first", "shared", "generation", 1, 10), ("reflect", "reflection", "reflection", 1, 20), ("second", "reflection", "generation", 2, 30)]:
        calls.append({"invocation_id": call_id, "model": "gpt-6-astra", "instance": "case", "mode": mode, "stage": stage, "attempt": attempt, "backend": "codex-cli", "classification": "shared_first" if mode == "shared" else "policy", "physical_call_confirmed": True, "usage_available": True, "usage": {"input_tokens": input_tokens, "output_tokens": 1, "cached_input_tokens": 0, "cache_write_input_tokens": 0, "reasoning_output_tokens": 0}})
    policies = {}
    for mode in ("plain", "reflection"):
        policies[mode] = {"state": "complete", "audit_issues": [], "call_ids": ["first"] if mode == "plain" else ["first", "reflect", "second"], "attempt_metrics": {str(k): {m: mode == "reflection" and k == 2 for m in ("execution", "verification", "quality")} for k in (1, 2)}}
    pair = {"model": "gpt-6-astra", "suite": "bd", "problem": "ufl", "instance": "case", "size": "small", "seed": 0, "state": "complete", "policies": policies}
    return {"audit_errors": [], "snapshot": {"read_errors": []}, "configuration": {"protocol_id": "fixture", "max_attempts": 2}, "cohort_manifest": {"instances": [pair]}, "input_root": "/fixture", "pairs": [pair], "calls": calls}


def test_reflection_buys_next_candidate_and_unsolved_case_stays_in_denominator():
    result = build_token_analysis([synthetic_report()], read_supplemental=False)
    by_key = {(r["mode"], r["budget"]): r for r in result["budget_records"]}
    assert by_key[("plain", 1)]["agent_accounted_tokens"] == by_key[("reflection", 1)]["agent_accounted_tokens"] == 11
    assert by_key[("reflection", 2)]["agent_accounted_tokens"] == 63
    assert len(result["calls"]) == 3
    assert all(r["quality_rate"] == 0 and r["total_cases"] == 1 for r in result["resource_curves"] if r["mode"] == "plain")


def test_unknown_success_cost_is_not_counted_as_zero_or_silently_dropped():
    report = deepcopy(synthetic_report())
    report["calls"][-1]["usage_available"] = False
    result = build_token_analysis([report], read_supplemental=False)
    assert all(r["quality_rate"] is None and r["unlocated_quality_successes"] == 1 and r["total_cases"] == 1 for r in result["resource_curves"] if r["mode"] == "reflection")


def test_supplement_reads_terminal_usage_once_and_preserves_source(tmp_path):
    usage = {"input_tokens": 2, "cache_creation_input_tokens": 5, "cache_read_input_tokens": 20, "output_tokens": 30}
    event = {"type": "result", "usage": usage | {"output_tokens_details": {"thinking_tokens": 17}}, "modelUsage": {"claude": {"thinkingTokens": 17}}, "duration_ms": 4500, "duration_api_ms": 4000, "num_turns": 3}
    folder = tmp_path / "call"
    folder.mkdir()
    path = folder / "agent_events.jsonl"
    original = json.dumps({"type": "assistant", "usage": usage}) + "\n" + json.dumps(event) + "\n"
    path.write_text(original)
    result = supplemental_claude({"backend": "claude-code-cli", "stage": "generation", "path": "call", "usage": usage, "invocation_id": "test"}, tmp_path)
    assert result["thinking_tokens"] == 17
    assert result["provider_duration_seconds"] == 4.5
    assert result["provider_api_seconds"] == 4
    assert result["num_turns"] == 3
    assert path.read_text() == original
