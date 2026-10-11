from datetime import datetime, timedelta, timezone
import json
import subprocess
from types import SimpleNamespace

import pytest

from src import hourly_review_scheduler as scheduler


def run_schedule(tmp_path, monkeypatch, deliveries, *, recurring=False, acknowledge=False, clock=None, send=None, sleep=None):
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("Inspect the saved experiment and preserve its outcomes.")
    schedule = {"thread_id": "current-thread", "codex_executable": "codex", "cwd": str(tmp_path), "prompt_file": str(prompt), "interval_seconds": 3600, "repeat_until_stopped": recurring, "require_review_ack": acknowledge, "deliveries": deliveries}
    path = tmp_path / "schedule.json"
    path.write_text(json.dumps(schedule))
    calls = []
    def invoke(command, **kwargs):
        calls.append(command)
        return send(command) if send else SimpleNamespace(returncode=0, stdout="queued", stderr="")
    monkeypatch.setattr(scheduler.subprocess, "run", invoke)
    monkeypatch.setattr(scheduler.signal, "signal", lambda *args: None)
    monkeypatch.setattr(scheduler, "now", lambda: clock[0])
    monkeypatch.setattr(scheduler.time, "sleep", sleep or (lambda seconds: (tmp_path / "STOP").write_text("finished")))
    scheduler.run(path)
    return json.loads(path.read_text()), json.loads((tmp_path / "scheduler_status.json").read_text()), calls


def test_bounded_schedule_sends_only_latest_overdue_review(tmp_path, monkeypatch):
    instant = datetime(2026, 10, 9, 12, tzinfo=timezone.utc)
    items = [{"index": index, "due_at": (instant - timedelta(hours=4-index)).isoformat(), "state": "pending"} for index in range(1, 4)]
    saved, status, calls = run_schedule(tmp_path, monkeypatch, items, clock=[instant])
    assert [item["state"] for item in saved["deliveries"]] == ["missed_superseded", "missed_superseded", "queued"]
    assert len(calls) == 1 and "3/3" in calls[0][-1]
    assert status["state"] == "deliveries_finished"


def test_recurring_review_keeps_hourly_cadence_after_sleep_and_obeys_stop(tmp_path, monkeypatch):
    instant = datetime(2026, 10, 9, 12, 25, tzinfo=timezone.utc)
    items = [{"index": 1, "due_at": instant.replace(hour=8, minute=0).isoformat(), "state": "pending"}]
    saved, status, calls = run_schedule(tmp_path, monkeypatch, items, recurring=True, clock=[instant])
    assert len(calls) == 1
    assert saved["deliveries"][1] == {"index": 2, "due_at": instant.replace(hour=13, minute=0).isoformat(), "state": "pending"}
    assert status["state"] == "stopped"


def test_unacknowledged_review_does_not_queue_a_second_turn(tmp_path, monkeypatch):
    instant = datetime(2026, 10, 9, 12, tzinfo=timezone.utc)
    items = [{"index": 1, "due_at": (instant - timedelta(hours=2)).isoformat(), "state": "queued"}, {"index": 2, "due_at": (instant - timedelta(hours=1)).isoformat(), "state": "pending"}]
    observed = []
    def sleep(seconds):
        status = json.loads((tmp_path / "scheduler_status.json").read_text())
        observed.append(status["state"])
        if status["state"] == "waiting_for_review" and status["review_index"] == 1:
            (tmp_path / "review_0001.json").write_text('{"review_completed":true}')
        else:
            (tmp_path / "STOP").write_text("finished")
    saved, status, calls = run_schedule(tmp_path, monkeypatch, items, recurring=True, acknowledge=True, clock=[instant], sleep=sleep)
    assert observed[0] == "waiting_for_review"
    assert len(calls) == 1 and "review_0002.json" in calls[0][-1]
    assert saved["deliveries"][1]["state"] == "queued"


@pytest.mark.parametrize("initial_state", ["sending", "pending"])
def test_restart_and_delivery_timeout_preserve_uncertainty_without_duplicate_retry(tmp_path, monkeypatch, initial_state):
    instant = datetime(2026, 10, 9, 12, tzinfo=timezone.utc)
    items = [{"index": 1, "due_at": instant.isoformat(), "state": initial_state}]
    def timeout(command):
        raise subprocess.TimeoutExpired(command, 45)
    saved, status, calls = run_schedule(tmp_path, monkeypatch, items, recurring=True, clock=[instant], send=timeout)
    expected = "delivery_unknown_after_restart" if initial_state == "sending" else "delivery_unknown"
    assert saved["deliveries"][0]["state"] == expected
    assert len(calls) == (0 if initial_state == "sending" else 1)
    assert saved["deliveries"][1]["due_at"] == (instant + timedelta(hours=1)).isoformat()


def test_existing_stop_prevents_any_recurring_delivery(tmp_path, monkeypatch):
    instant = datetime(2026, 10, 9, 12, tzinfo=timezone.utc)
    (tmp_path / "STOP").write_text("User paused the campaign")
    items = [{"index": 1, "due_at": instant.isoformat(), "state": "pending"}]
    saved, status, calls = run_schedule(tmp_path, monkeypatch, items, recurring=True, clock=[instant])
    assert calls == [] and status["state"] == "stopped"
