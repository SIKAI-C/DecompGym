"""Normalize pinned CLI event streams while retaining raw backend usage fields."""
from __future__ import annotations

import json


def parse_events(backend, contents, *, returncode, requested_model):
    result = {"terminal_state": None, "usage": {}, "usage_available": False, "error": None, "warnings": [], "observed_models": [], "response": ""}
    observed = set()
    for line in contents.splitlines():
        try:
            event = json.loads(line)
        except (ValueError, RecursionError):
            continue
        if not isinstance(event, dict):
            continue
        if backend == "codex-cli":
            model = event.get("model")
            if isinstance(model, str) and model.strip():
                observed.add(model)
            if event.get("type") == "turn.completed":
                result["terminal_state"] = "completed"
                if returncode == 0:
                    result["error"] = None
                usage = event.get("usage")
                if isinstance(usage, dict):
                    result["usage_available"] = True
                    for key, value in usage.items():
                        if type(value) is int:
                            result["usage"][key] = result["usage"].get(key, 0) + value
            elif event.get("type") in {"error", "turn.failed"}:
                payload = event.get("error") or event
                message = str(payload.get("message", payload)) if isinstance(payload, dict) else str(payload)
                if event["type"] == "turn.failed":
                    result.update(terminal_state="failed", error=message)
                else:
                    result["warnings"].append(message)
        elif backend == "claude-code-cli":
            if event.get("type") == "assistant":
                message = event.get("message")
                model = message.get("model") if isinstance(message, dict) else None
                if isinstance(model, str) and model:
                    observed.add(model)
            if event.get("type") == "result":
                result["terminal_state"] = "completed" if event.get("subtype") == "success" and not event.get("is_error") else "failed"
                usage = event.get("usage")
                result["usage_available"] = isinstance(usage, dict)
                result["usage"] = {key: value for key, value in (usage or {}).items() if type(value) is int} if isinstance(usage, dict) else {}
                if isinstance(event.get("modelUsage"), dict):
                    observed.update(key for key in event["modelUsage"] if isinstance(key, str) and key)
                result["response"] = str(event.get("result") or "")
                if result["terminal_state"] != "completed":
                    result["error"] = str(event.get("errors") or event.get("result") or event.get("subtype"))
        elif backend == "agy-cli":
            if event.get("event") == "result" and isinstance(event.get("result"), dict):
                payload = event["result"]
                result["terminal_state"] = "completed" if payload.get("status") == "SUCCESS" else "failed"
                usage = payload.get("usage")
                result["usage_available"] = isinstance(usage, dict)
                result["usage"] = {key: value for key, value in (usage or {}).items() if type(value) is int} if isinstance(usage, dict) else {}
                result["error"] = str(payload["error"]) if payload.get("error") else None
                result["response"] = str(payload.get("response") or "")
        else:
            raise ValueError(f"Unsupported CLI backend: {backend}")
    result["observed_models"] = sorted(observed)
    if backend == "claude-code-cli" and result["terminal_state"] == "completed" and (not observed or any(model != requested_model and not model.startswith(requested_model + "-") for model in observed)):
        result.update(terminal_state="infrastructure_failed", error=f"Claude Code model identity unavailable or mismatched: requested {requested_model}, observed {sorted(observed)}")
    return result
