"""Shared test isolation: auth-volume locks live in a per-test directory, never in the machine's runtime state."""
import pytest


@pytest.fixture(autouse=True)
def private_auth_locks(tmp_path_factory, monkeypatch):
    from src.agent import auth
    directory = tmp_path_factory.mktemp("auth-locks")
    monkeypatch.setattr(auth, "lock_directory", lambda: directory)
    return directory
