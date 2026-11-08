"""Bound concurrent agent calls per backend and fall back to one call after an authentication failure.

Each slot is bound to its own auth volume (src.agent.auth.SLOT_VOLUMES: Codex, and Claude Code in the lanes variant):
`enter()` yields the slot index, and the campaign mounts that slot's volume for the call. The limit-1 fallback is the
single-volume mode: it holds the backend's legacy lock exclusively and always yields slot 0. A campaign creates one
gate per backend and shares it by every lane and the method judge.

Every wait (the in-process admission, the legacy lock, the slot locks) gives up with InfrastructureError as soon as
the controller is asked to stop (src.agent.auth.CANCEL), so a stop never hangs at a gate.
"""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import os
import threading
import time

from src.utils.errors import InfrastructureError
from src.utils.records import now, save


# Per backend: the legacy lock name (the serialized runners' exclusive lock, kept), the fallback record in the campaign
# root and its limit field. Codex keeps the names of the two-slot gate, so its records are unchanged.
GATES = {"codex-cli": ("codex-native-auth", "auth_fallback.json", "effective_codex_invocation_limit"),
         "claude-code-cli": ("native-claude-auth", "auth_fallback.claude-code-cli.json", "effective_claude_invocation_limit")}
POLL_SECONDS = 0.05


def require_running(what):
    from src.agent import auth
    if auth.CANCEL.is_set():
        raise InfrastructureError(f"Stopped while waiting for {what} (stop requested)")


def acquire(descriptor, operation, what):
    """flock without blocking in the kernel, so that a stop request is observed while waiting."""
    while True:
        try:
            fcntl.flock(descriptor, operation | fcntl.LOCK_NB)
            return
        except BlockingIOError:
            require_running(what)
            time.sleep(POLL_SECONDS)


@contextmanager
def exclusive_lock(state_directory, name):
    """One call at a time for a backend without slots (AGY; Claude Code outside the lanes variant): the exclusive lock
    `<name>.lock` in the private state directory, the file src.experiments.locks.runtime_lock uses, but cancellable."""
    descriptor = os.open(state_directory / f"{name}.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        acquire(descriptor, fcntl.LOCK_EX, f"the {name} lock")
        require_running(f"the {name} lock")
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


class NativeCallGate:
    """`slots` shared-lock slots, or one exclusive legacy lock after an auth failure; `enter()` yields the slot index."""
    def __init__(self, state_directory, phase_directory, *, backend="codex-cli", slots=2):
        from src.agent.auth import SLOT_VOLUMES
        if backend not in GATES or not isinstance(slots, int) or not 1 <= slots <= len(SLOT_VOLUMES[backend]):
            raise ValueError(f"A {backend} call gate has between 1 and {len(SLOT_VOLUMES.get(backend, ()))} slots, one per auth volume: {slots!r}")
        self.state_directory = state_directory
        self.phase_directory = phase_directory
        self.backend, self.slots = backend, slots
        self.lock_name, fallback_name, self.limit_field = GATES[backend]
        self.fallback_path = phase_directory / fallback_name
        self.condition = threading.Condition()
        self.active = 0

    @property
    def limit(self):
        return 1 if self.fallback_path.exists() else self.slots

    def fallback(self, marker, *, role, invocation_id):
        with self.condition:
            if not self.fallback_path.exists():
                save(self.fallback_path, {"recorded_at": now(), self.limit_field: 1, "reason": marker, "role": role, "invocation_id": invocation_id, "policy": "drain admitted calls, then serialize; do not change credentials"})
            self.condition.notify_all()

    def descriptor(self, name):
        return os.open(self.state_directory / name, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)

    @contextmanager
    def enter(self):
        what = f"a {self.backend} call slot"
        with self.condition:
            while self.active >= self.limit:
                require_running(what)
                self.condition.wait(timeout=0.2)
            require_running(what)
            parallel = self.limit > 1
            self.active += 1
        legacy, slot, index = None, None, 0
        try:
            legacy = self.descriptor(f"{self.lock_name}.lock")
            # Existing serialized runners use this same lock exclusively.
            acquire(legacy, fcntl.LOCK_SH if parallel else fcntl.LOCK_EX, what)
            if parallel:
                while slot is None:
                    for candidate_index in range(self.slots):
                        candidate = self.descriptor(f"{self.lock_name}.slot{candidate_index}.lock")
                        try:
                            fcntl.flock(candidate, fcntl.LOCK_EX | fcntl.LOCK_NB)
                            slot, index = candidate, candidate_index
                            break
                        except BlockingIOError:
                            os.close(candidate)
                    if slot is None:
                        require_running(what)
                        time.sleep(POLL_SECONDS)
            require_running(what)  # the last check before the call is admitted
            yield index
        finally:
            for descriptor in (slot, legacy):
                if descriptor is not None:
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
                    os.close(descriptor)
            with self.condition:
                self.active -= 1
                self.condition.notify_all()


def auth_failure_marker(result, stderr):
    text = " ".join([result.error or "", *result.warnings, stderr]).lower()
    markers = ("refresh_token_reused", "refresh token", "token refresh failed", "failed to refresh access token", "failed to refresh authentication", "token has already been used", "unauthorized", "authentication failed", "authentication required", "access token has expired", "not logged in", "401 unauthorized", "status code 401", "http 401")
    return next((marker for marker in markers if marker in text), None)
