"""Deliver bounded or recurring reviews into an existing Codex conversation."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def now():
    return datetime.now(timezone.utc)


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def add_recurring_delivery(schedule, instant):
    """Keep one future delivery, advancing past missed slots without a backlog."""
    if not schedule.get("repeat_until_stopped") or any(item["state"] == "pending" for item in schedule["deliveries"]):
        return False
    interval = timedelta(seconds=schedule["interval_seconds"])
    if interval.total_seconds() <= 0:
        raise ValueError("Recurring review interval must be positive")
    previous = schedule["deliveries"][-1]
    due = datetime.fromisoformat(previous["due_at"]) + interval
    if due <= instant:
        due += interval * ((instant - due) // interval + 1)
    schedule["deliveries"].append({"index": previous["index"] + 1, "due_at": due.isoformat(), "state": "pending"})
    return True


def run(schedule_path):
    directory = schedule_path.parent
    status_path = directory / "scheduler_status.json"
    stopped = False
    def stop(signum, frame):
        nonlocal stopped
        stopped = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    with (directory / "scheduler.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("This hourly review scheduler is already running")
        schedule = json.loads(schedule_path.read_text())
        for item in schedule["deliveries"]:
            if item["state"] == "sending":
                item["state"] = "delivery_unknown_after_restart"
        save(schedule_path, schedule)
        while not stopped and not (directory / "STOP").exists():
            instant = now()
            if add_recurring_delivery(schedule, instant):
                save(schedule_path, schedule)
            pending = [item for item in schedule["deliveries"] if item["state"] == "pending"]
            if not pending:
                break
            if schedule.get("require_review_ack"):
                outstanding = [item for item in schedule["deliveries"] if item["state"] == "queued" and not (directory / f"review_{item['index']:04d}.json").is_file()]
                if outstanding:
                    save(status_path, {"state": "waiting_for_review", "pid": os.getpid(), "checked_at": instant.isoformat(), "review_index": outstanding[-1]["index"], "thread_id": schedule["thread_id"]})
                    time.sleep(30)
                    continue
            save(status_path, {"state": "waiting", "pid": os.getpid(), "checked_at": instant.isoformat(), "next_review_at": pending[0]["due_at"], "remaining": len(pending), "thread_id": schedule["thread_id"]})
            due = [item for item in pending if datetime.fromisoformat(item["due_at"]) <= instant]
            if due:
                # If the machine was unavailable, request one current review instead of flooding the chat.
                for item in due[:-1]:
                    item.update(state="missed_superseded", updated_at=instant.isoformat())
                item = due[-1]
                item.update(state="sending", sent_at=instant.isoformat())
                save(schedule_path, schedule)
                prompt = Path(schedule["prompt_file"]).read_text()
                label = schedule.get("label", "User-authorized overnight DecompBench review")
                number = str(item["index"]) if schedule.get("repeat_until_stopped") else f"{item['index']}/{len(schedule['deliveries'])}"
                acknowledgement_path = directory / f"review_{item['index']:04d}.json"
                acknowledgement = f"\nAfter completing this review, save its evidence to {acknowledgement_path} so the next hourly review can be delivered.\n" if schedule.get("require_review_ack") else ""
                message = f"[{label} {number}; scheduled {item['due_at']}]\n{acknowledgement}\n{prompt}"
                try:
                    result = subprocess.run([schedule["codex_executable"], "queue", "--thread", schedule["thread_id"], "--message", message], cwd=schedule["cwd"], capture_output=True, text=True, timeout=45)
                    item.update(state="queued" if result.returncode == 0 else "delivery_failed", returncode=result.returncode, receipt=result.stdout[-4000:], error=result.stderr[-2000:])
                except subprocess.TimeoutExpired:
                    item.update(state="delivery_unknown", error="CLI timed out; delivery may have succeeded, so no duplicate retry was sent")
                except OSError as error:
                    item.update(state="delivery_failed", error=f"{type(error).__name__}: {error}")
                item["finished_at"] = now().isoformat()
                save(schedule_path, schedule)
                with (directory / "deliveries.jsonl").open("a") as stream:
                    stream.write(json.dumps(item) + "\n")
                continue
            time.sleep(min(30, max(0.1, (datetime.fromisoformat(pending[0]["due_at"]) - instant).total_seconds())))
        save(status_path, {"state": "stopped" if stopped or (directory / "STOP").exists() else "deliveries_finished", "pid": os.getpid(), "updated_at": now().isoformat(), "thread_id": schedule["thread_id"], "queued": sum(item["state"] == "queued" for item in schedule["deliveries"]), "failed_or_unknown": sum(item["state"].startswith("delivery_") for item in schedule["deliveries"])})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", type=Path, required=True)
    run(parser.parse_args().schedule.resolve())
