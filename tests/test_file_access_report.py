"""scripts/file_access_report.py counts a mounted file as opened only from successful tool calls naming it as one token."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("file_access_report", ROOT / "scripts/file_access_report.py")
report = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(report)


def claude_events(*calls):
    """An assistant tool_use and its tool_result per (id, input, is_error)."""
    lines = []
    for identifier, value, error in calls:
        lines.append({"type": "assistant", "message": {"content": [{"type": "tool_use", "id": identifier, "name": "Read", "input": value}]}})
        if error is not None:
            lines.append({"type": "user", "message": {"content": [{"type": "tool_result", "tool_use_id": identifier, "is_error": error, "content": "x"}]}})
    return "\n".join(json.dumps(line) for line in lines)


def test_a_claude_tool_use_counts_only_with_a_successful_result():
    events = claude_events(("ok", {"file_path": "/work/history/attempt_01/solver.py"}, False), ("failed", {"file_path": "/work/history/attempt_01/debug.md"}, True),
                           ("unanswered", {"file_path": "/work/instance/data.json"}, None))
    inputs = report.tool_inputs(events)
    assert report.opened("/work/history/attempt_01/solver.py", inputs)
    assert not report.opened("/work/history/attempt_01/debug.md", inputs) and not report.opened("/work/instance/data.json", inputs)


def test_a_path_matches_only_as_one_token():
    separate = ["cat history/attempt_01/solver.py history/attempt_02/debug.md"]
    assert report.opened("/work/history/attempt_01/solver.py", separate) and report.opened("/work/history/attempt_02/debug.md", separate)
    assert not report.opened("/work/history/attempt_01/debug.md", separate)  # a name and a directory mentioned apart
    assert not report.opened("/work/instance/data.json", ["cd /work/instance && cat data.json.bak"])
    assert not report.opened("/work/history/attempt_01/solver.py", ["cat history/attempt_011/solver.py", "cat attempt_01/solver.py.orig"])
    assert report.opened("/work/history/attempt_02/debug.md", ["for f in history/attempt_{01..03}/debug.md; do cat $f; done"])
    assert report.opened("/work/history/attempt_03/debug.md", ["cat /work/history/attempt_*/debug.md"]) and report.opened("/work/instance/data.json", ['{"file_path": "/work/instance/data.json"}'])
    assert "estimate" in report.__doc__ and "not verified read coverage" in report.__doc__


def test_brace_expansion_never_materializes_beyond_its_budget():
    import time
    started = time.monotonic()
    assert len(report.expand_braces("history/attempt_{1..1000000000}/debug.md")) == 256
    assert len(report.expand_braces("{1..1000000000}{1..1000000000}{a,b}", limit=10)) == 10
    assert report.expand_braces("x{01..03}y{a,b}", limit=4) == ["x01ya", "x01yb", "x02ya", "x02yb"]
    assert report.expand_braces("{5..3}") == ["5", "4", "3"] and report.expand_braces("{a,b}", limit=0) == []
    assert time.monotonic() - started < 1


def test_antigravity_tool_steps_count_once_done_without_error():
    done = json.dumps({"event": "step_update", "step_update": {"step_type": "tool", "state": "DONE", "tool_name": "view_file",
                                                               "tool_info": {"name": "view_file", "parameters": {"AbsolutePath": "/work/instance/data.json"}}}})
    active = json.dumps({"event": "step_update", "step_update": {"step_type": "tool", "state": "ACTIVE", "tool_name": "view_file",
                                                                 "tool_info": {"name": "view_file", "parameters": {"AbsolutePath": "/work/history/attempt_01/debug.md"}}}})
    inputs = report.tool_inputs(done + "\n" + active)
    assert report.opened("/work/instance/data.json", inputs) and not report.opened("/work/history/attempt_01/debug.md", inputs)
