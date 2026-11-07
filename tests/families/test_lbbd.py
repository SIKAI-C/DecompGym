"""Assignment and scheduling LBBD (small scale): verifier regressions."""
from __future__ import annotations




def test_lbbd_verifier_rejects_overflowing_starts():
    from src.bd.lbbd.d_verifier import verify_solution

    data = {"n_tasks": 1, "n_machines": 1, "release": [0], "due": [1], "weight": [2], "proc_time": [[1]], "proc_cost": [[1]]}
    assert not verify_solution(data, {"solution": {"assignments": [{"task_id": 0, "machine_id": 0, "start": 1e308}]}})["ok"]
    assert not verify_solution(data, {"solution": {"assignments": [{"task_id": 0, "machine_id": 0, "start": 10**400}]}})["ok"]
