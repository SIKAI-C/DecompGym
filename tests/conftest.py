"""Shared test isolation: auth-volume locks live in a per-test directory, never in the machine's runtime state; each test
starts with an open native-solver admission and no stop request."""
import pytest


@pytest.fixture(autouse=True)
def private_auth_locks(tmp_path_factory, monkeypatch):
    from src.agent import auth
    directory = tmp_path_factory.mktemp("auth-locks")
    monkeypatch.setattr(auth, "lock_directory", lambda: directory)
    return directory


@pytest.fixture(autouse=True)
def fresh_controller_latches(monkeypatch):
    from src.agent import auth
    from src.experiments import sandbox
    monkeypatch.setattr(sandbox, "SOLVER_ADMISSION", sandbox.SolverAdmission())
    auth.CANCEL.clear()
    yield
    auth.CANCEL.clear()
