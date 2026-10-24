"""Bound concurrent Codex calls and fall back to one call after an authentication failure."""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import os
import threading
import time

from src.utils.records import now, save


class NativeCallGate:
    """Two shared-lock slots, or one exclusive legacy lock after an auth failure."""
    def __init__(self, state_directory, phase_directory):
        self.state_directory = state_directory
        self.phase_directory = phase_directory
        self.fallback_path = phase_directory / "auth_fallback.json"
        self.condition = threading.Condition()
        self.active = 0

    @property
    def limit(self):
        return 1 if self.fallback_path.exists() else 2

    def fallback(self, marker, *, role, invocation_id):
        with self.condition:
            if not self.fallback_path.exists():
                save(self.fallback_path, {"recorded_at": now(), "effective_codex_invocation_limit": 1, "reason": marker, "role": role, "invocation_id": invocation_id, "policy": "drain admitted calls, then serialize; do not change credentials"})
            self.condition.notify_all()

    def descriptor(self, name):
        return os.open(self.state_directory / name, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)

    @contextmanager
    def enter(self):
        with self.condition:
            while self.active >= self.limit:
                self.condition.wait(timeout=0.2)
            parallel = self.limit == 2
            self.active += 1
        legacy, slot = None, None
        try:
            legacy = self.descriptor("codex-native-auth.lock")
            # Existing serialized runners use this same lock exclusively.
            fcntl.flock(legacy, fcntl.LOCK_SH if parallel else fcntl.LOCK_EX)
            if parallel:
                while slot is None:
                    for index in range(2):
                        candidate = self.descriptor(f"codex-native-auth.slot{index}.lock")
                        try:
                            fcntl.flock(candidate, fcntl.LOCK_EX | fcntl.LOCK_NB)
                            slot = candidate
                            break
                        except BlockingIOError:
                            os.close(candidate)
                    if slot is None:
                        time.sleep(0.05)
            yield
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
