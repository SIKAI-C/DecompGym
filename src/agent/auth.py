"""Host side of the Docker agents' authentication plane: auth volumes, per-volume locks and terminal auth failures.

The trusted worker (docker/agents/worker.py) and the agent image are unchanged (their hashes are pinned by
tests/test_single_file_golden.py): the worker prefers the credential saved in its auth volume and writes refreshed
Codex tokens back when the CLI exits. Everything else is fixed here, on the host:

- Source. The auth volume is the only credential source of Codex and Claude Code: the job carries no credential
  (`credential_from_environment` returns None for both) and a credential enters a volume only through
  `docker auth --login` (src/experiments/agent_auth.py). AGY receives only its keyring passphrase.
- Volumes. Codex has one auth volume per call-gate slot (CODEX_SLOT_VOLUMES), each with its own device sign-in, so two
  running Codex containers never share one OAuth refresh lineage. Slot 0 is the single-volume mode: the call gate's
  limit-1 fallback and every call outside a campaign use it. Claude Code has one volume per slot as well
  (CLAUDE_SLOT_VOLUMES; the same `claude setup-token` token may be stored in each): only an execution variant with two
  Claude calls at a time (src.experiments.execution_policy.LANES_EXECUTION) uses slot 1.
- Locks. `volume_lock(volume)` is one exclusive host lock per auth volume (a file in the private runtime state
  directory shared by all worktrees). It is held for every agent call, recovery, login, check and verify that mounts
  the volume, and released on exception, timeout and cancel. Acquiring it again in the thread that holds it raises
  instead of deadlocking. Lock order: a campaign's call gate or backend lock first, then the volume lock. Because a
  lock dies with its holder (a crash, a failed container cleanup), every holder then calls
  `require_exclusive_volume`: a running container that still mounts the volume outlived its owner, so it is waited
  for, removed if it is an orphaned auth helper, and otherwise refused.
- Job credentials. `forbid_job_credential` refuses a Codex or Claude Code job that carries a credential the worker
  would use, so no caller can bypass the volume.
- Terminal failures. `terminal_auth_marker` recognizes a backend's terminal credential failure in the CLI's stderr and
  in its error events of an unsuccessful call, never in generated text.
"""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import re
import threading
import time

from src.experiments.locks import state_root


AUTH_POLICY = "volume-only-host-locked-v1"
# Slot 0 keeps the established Codex volume name (so the single-file paper path's records are unchanged); slots 1 and 2
# are later (slot 2 only for the three Codex calls of the lanes variant).
CODEX_SLOT_VOLUMES = ("decompbench-v3-codex-auth", "decompbench-v3-codex-auth-slot1", "decompbench-v3-codex-auth-slot2")
# Slot 0 is the established Claude Code volume; slot 1 is used only by the lanes variant's second Claude call.
CLAUDE_SLOT_VOLUMES = ("decompbench-v2-claude-auth", "decompbench-v2-claude-auth-slot1")
AUTH_VOLUMES = {"codex-cli": CODEX_SLOT_VOLUMES[0], "claude-code-cli": CLAUDE_SLOT_VOLUMES[0], "agy-cli": "decompbench-agy-native-keyring-v1"}
# The backends whose calls are bound to call-gate slots, each slot with its own auth volume.
SLOT_VOLUMES = {"codex-cli": CODEX_SLOT_VOLUMES, "claude-code-cli": CLAUDE_SLOT_VOLUMES}
MOUNT_POINTS = {"codex-cli": "/auth", "claude-code-cli": "/auth", "agy-cli": "/auth/keyrings"}
VOLUME_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
HELPER_LABEL = "v2-auth-helper"  # io.decompbench.runtime label of the one-off login/check containers
ORPHAN_WAIT_SECONDS = 45  # a worker whose owner died stops within seconds (heartbeat), then checkpoints and exits
LOCK_WAIT_SECONDS = 1800  # twice the longest agent call; a longer wait (for example an open interactive login) is retryable
# Set when the controller is asked to stop: a thread waiting for a volume lock gives up at once.
CANCEL = threading.Event()
# Job credential keys the pinned worker would use for each backend (docker/agents/worker.py credentials()).
USABLE_JOB_CREDENTIAL = {"codex-cli": ("auth_mode", "tokens", "OPENAI_API_KEY"), "claude-code-cli": ("oauth_token", "credentials")}
REPAIR = {"codex-cli": "python src/02_experiment.py docker auth --backend codex --slot {slot} --login --replace",
          "claude-code-cli": "python src/02_experiment.py docker auth --backend claude --login --replace",
          "agy-cli": "python src/02_experiment.py docker agy-login"}


def auth_volume(backend: str, slot: int | None = None) -> str:
    """The auth volume of a backend; for Codex and Claude Code, the volume of call-gate slot `slot` (None: slot 0)."""
    if backend not in SLOT_VOLUMES:
        return AUTH_VOLUMES[backend]
    volumes, index = SLOT_VOLUMES[backend], 0 if slot is None else slot
    if not isinstance(index, int) or not 0 <= index < len(volumes):
        raise ValueError(f"Unknown {backend} auth slot: {slot!r}")
    return volumes[index]


def backend_volumes(backend: str) -> tuple[str, ...]:
    return SLOT_VOLUMES.get(backend, (AUTH_VOLUMES[backend],))


def volume_slot(backend: str, volume: str) -> int | None:
    """The call-gate slot of an auth volume (None outside the backend's slots). Claude Code's slot 0 is its single
    established volume and stays None, as in every record written before Claude Code had a second slot."""
    volumes = SLOT_VOLUMES.get(backend, ())
    if volume not in volumes:
        return None
    index = volumes.index(volume)
    return None if backend == "claude-code-cli" and index == 0 else index


def repair_command(backend: str, volume: str | None = None) -> str:
    slot = volume_slot(backend, volume) if volume else None
    if backend == "claude-code-cli" and slot is not None:
        return REPAIR[backend].replace("--backend claude", f"--backend claude --slot {slot}")
    return REPAIR[backend].format(slot=0 if slot is None else slot)


def mounted_auth_volume(metadata: dict) -> str | None:
    """The auth volume recorded in a container state's mounts (None before the container existed)."""
    for mount in (metadata or {}).get("mounts") or []:
        destination, name = mount.get("Destination") or "", mount.get("Name") or ""
        if mount.get("Type") == "volume" and (destination == "/auth" or destination.startswith("/auth/")) and VOLUME_NAME.fullmatch(name):
            return name
    return None


_GUARD = threading.Lock()
_HELD: set[tuple[str, int]] = set()


def lock_directory() -> Path:
    return state_root()


def held_volumes() -> set[str]:
    with _GUARD:
        return {volume for volume, _ in _HELD}


@contextmanager
def volume_lock(volume: str, *, waiting=None, timeout: float = LOCK_WAIT_SECONDS):
    """Hold the exclusive host lock of one auth volume; `waiting(volume)` is called once if another holder blocks it.

    The wait is bounded and cancellable (CANCEL): it raises InfrastructureError, so a campaign retries later."""
    if not isinstance(volume, str) or not VOLUME_NAME.fullmatch(volume):
        raise ValueError(f"Invalid auth volume name: {volume!r}")
    key = (volume, threading.get_ident())
    with _GUARD:
        if key in _HELD:
            # flock on a second descriptor of the same file would wait for this thread forever.
            from src.experiments.environment import RuntimeEnvironmentError
            raise RuntimeEnvironmentError(f"Nested lock of auth volume {volume} in one thread")
    descriptor = os.open(lock_directory() / f"auth-volume.{volume}.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        deadline, notified = time.monotonic() + timeout, False
        while True:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if waiting is not None and not notified:
                    waiting(volume)
                    notified = True
                if CANCEL.is_set() or time.monotonic() >= deadline:
                    from src.utils.errors import InfrastructureError
                    raise InfrastructureError(f"Auth volume {volume} stayed locked by another call, check or login" + (" (stop requested)" if CANCEL.is_set() else f" for {timeout:.0f} s")) from None
                time.sleep(0.5)
        with _GUARD:
            _HELD.add(key)
        try:
            yield volume
        finally:
            with _GUARD:
                _HELD.discard(key)
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def require_exclusive_volume(engine, volume: str, *, allowed=(), wait: float | None = None) -> None:
    """With the volume lock held, no other running container may use `volume`.

    Every legitimate user holds the lock while its container runs, so such a container outlived its owner (host crash,
    failed cleanup). It is waited for, then removed if it is an orphaned auth helper; any other survivor is refused.
    `allowed` names the container being recovered. An engine without `volume_users` (a test double) is not probed."""
    probe = getattr(engine, "volume_users", None)
    if probe is None:
        return
    deadline = time.monotonic() + (ORPHAN_WAIT_SECONDS if wait is None else wait)
    while True:
        users = [(name, label) for name, label in probe(volume) if name not in allowed]
        if not users or time.monotonic() >= deadline:
            break
        time.sleep(1)
    for name, label in users:
        if label == HELPER_LABEL:
            engine.remove_container(name)
    remaining = sorted(name for name, _ in probe(volume) if name not in allowed)
    if remaining:
        from src.experiments.environment import RuntimeEnvironmentError
        raise RuntimeEnvironmentError(f"Auth volume {volume} is still used by running container(s) {remaining} that hold no lock; stop them (docker rm --force <name>) and retry")


def forbid_job_credential(backend: str, credential) -> None:
    """The auth volume is the only Codex and Claude Code source: refuse a job credential the worker would use."""
    if isinstance(credential, dict) and any(key in credential for key in USABLE_JOB_CREDENTIAL.get(backend, ())):
        from src.experiments.environment import RuntimeEnvironmentError
        raise RuntimeEnvironmentError(f"A {backend} job may not carry a credential; store it in the auth volume with `docker auth --login`")


# Terminal credential failures of the pinned CLIs (codex-cli 0.162.1, Claude Code 2.1.296, agy 1.3.1), taken from the
# CLIs' own messages. They are matched lower-cased and only for a call that did not complete successfully.
CODEX_MARKERS = ("your access token could not be refreshed", "your authentication session could not be refreshed", "please log out and sign in again",
                 "refresh_token_reused", "refresh_token_expired", "refresh_token_invalidated", "invalidated oauth token", "token_invalidated",
                 "token has been invalidated", "provided authentication token is expired")
CODEX_EVENT_MARKERS = ("unexpected status 401", "401 unauthorized")  # an HTTP 401 counts only in a turn.failed or error event
CLAUDE_ERROR_KINDS = ("authentication_failed", "oauth_org_not_allowed")  # the structured `error` of a synthetic API-error message
CLAUDE_MARKERS = ("authentication_error", "oauth token revoked", "oauth token has expired", "login expired", "please run /login", "invalid api key",
                  "invalid auth token", "oauth session expired", "failed to authenticate")
AGY_MARKERS = ("please sign in", "currently not signed in", "not signed in", "unauthenticated")


def _events(contents: str):
    for line in (contents or "").splitlines():
        try:
            event = json.loads(line)
        except (ValueError, RecursionError):
            continue
        if isinstance(event, dict):
            yield event


def _text(value) -> str:
    return value if isinstance(value, str) else json.dumps(value, sort_keys=True, default=str)


def terminal_events(backend: str, events: str):
    """(present, [(kind, text)]): whether the stream has a terminal event, and the texts of its terminal errors only.

    Never an agent message or tool output, and never an earlier error the CLI recovered from. Codex: the last
    turn.failed, else the last error event, both only after the last turn.completed. Claude Code: the last assistant
    message if it is an API-error message, and an error result. AGY: the last result."""
    found = {}
    for event in _events(events):
        if backend == "codex-cli" and event.get("type") == "turn.completed":
            found = {"turn.completed": True}  # the turn completed: earlier errors were recovered from
        elif backend == "codex-cli" and event.get("type") in {"error", "turn.failed"}:
            payload = event.get("error") or event
            found[event["type"]] = _text(payload.get("message", payload) if isinstance(payload, dict) else payload)
        elif backend == "claude-code-cli":
            if event.get("type") == "assistant":
                found.pop("assistant_error", None)  # a later message means the CLI went on after an earlier error
            if event.get("type") == "assistant" and isinstance(event.get("error"), str):
                message = event.get("message")
                content = message.get("content") if isinstance(message, dict) else None
                text = " ".join(_text(block.get("text", "")) for block in (content if isinstance(content, list) else []) if isinstance(block, dict) and block.get("type") == "text")
                found["assistant_error"] = (event["error"], text)
            elif event.get("type") == "result":
                found["result"] = _text(event.get("result") or "") + " " + _text(event.get("errors") or "") if event.get("is_error") else None
        elif backend == "agy-cli" and event.get("event") == "result" and isinstance(event.get("result"), dict):
            payload = event["result"]
            found["result"] = _text(payload["error"]) if payload.get("status") != "SUCCESS" and payload.get("error") else None
    texts = []
    if backend == "codex-cli":
        if "turn.failed" in found or "error" in found:
            texts.append(("event", found.get("turn.failed", found.get("error"))))
    elif backend == "claude-code-cli":
        if "assistant_error" in found:
            kind, text = found["assistant_error"]
            texts += [("claude_error", kind), ("event", text)]
        if found.get("result"):
            texts.append(("event", found["result"]))
    elif found.get("result"):
        texts.append(("event", found["result"]))
    return bool(found), texts


def terminal_auth_marker(backend: str, events: str, stderr: str, *, succeeded: bool) -> str | None:
    """The terminal credential-failure marker of an unsuccessful call, or None. A terminal event decides on its own;
    stderr is read only when the CLI ended without one (an earlier, recovered error in stderr is no verdict)."""
    if succeeded:
        return None
    markers = {"codex-cli": CODEX_MARKERS, "claude-code-cli": CLAUDE_MARKERS, "agy-cli": AGY_MARKERS}[backend]
    present, texts = terminal_events(backend, events)
    for kind, text in texts:
        if kind == "claude_error":
            if text in CLAUDE_ERROR_KINDS:
                return text
            continue
        lowered = text.lower()
        for marker in markers + (CODEX_EVENT_MARKERS if backend == "codex-cli" else ()):
            if marker in lowered:
                return marker
    if present:
        return None
    lowered = (stderr or "").lower()
    return next((marker for marker in markers if marker in lowered), None)
