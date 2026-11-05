"""Exact route costs over customer subsets, for column generation on small capacitated routing instances.

`RouteTable` runs a Held-Karp dynamic program from one start depot over every customer subset whose
demand fits the vehicle capacity: dp[S, j] is the cheapest path depot -> ... -> j visiting exactly S.
A subset's route cost is min_j dp[S, j] + (j -> depot). Pricing a column then costs one matrix-vector
product (cost - members @ prize), so pricing is exact, and enumerating every subset whose reduced cost is
within the integrality gap is a vectorized filter. Customers are 0..n-1; costs are integers.
"""
from __future__ import annotations

import numpy as np


INF = np.int64(1) << 40


class RouteTable:
    def __init__(self, start: np.ndarray, between: np.ndarray, end: np.ndarray, demand: np.ndarray, capacity: int) -> None:
        n = len(demand)
        start, between, end = (np.asarray(a, dtype=np.int64) for a in (start, between, end))
        demand = np.asarray(demand, dtype=np.int64)
        everything = np.arange(1 << n, dtype=np.int64)
        load = np.zeros(1 << n, dtype=np.int64)
        size = np.zeros(1 << n, dtype=np.int64)
        for j in range(n):
            bit = (everything >> j) & 1
            load += bit * demand[j]
            size += bit
        feasible = everything[(load <= capacity) & (size > 0)]
        feasible = feasible[np.argsort(size[feasible], kind="stable")]
        dp = np.full((1 << n, n), INF, dtype=np.int64)
        for j in range(n):
            if demand[j] <= capacity:
                dp[1 << j, j] = start[j]
        sizes = size[feasible]
        for k in range(1, int(sizes.max(initial=0))):
            layer = feasible[sizes == k]
            for j in range(n):
                prev = layer[((layer >> j) & 1) == 0]
                target = prev | (1 << j)
                keep = load[target] <= capacity
                prev, target = prev[keep], target[keep]
                if len(prev):
                    dp[target, j] = np.minimum(dp[target, j], (dp[prev] + between[:, j]).min(axis=1))
        self.n, self.start, self.between, self.end, self.dp = n, start, between, end, dp
        self.masks = feasible
        self.cost = (dp[feasible] + end).min(axis=1)
        self.members = ((feasible[:, None] >> np.arange(n)) & 1).astype(np.float64)
        self.load = load[feasible]
        self.index = {int(s): i for i, s in enumerate(feasible)}

    def reduced_costs(self, prize: np.ndarray, fixed: float = 0.0) -> np.ndarray:
        return self.cost + fixed - self.members @ np.asarray(prize, dtype=np.float64)

    def path(self, column: int) -> tuple[int, ...]:
        """Customers of column `column` in an optimal visiting order."""
        mask = int(self.masks[column])
        dp, between = self.dp, self.between
        j = int(np.argmin(dp[mask] + self.end))
        order = [j]
        while mask & (mask - 1):
            value, prev = dp[mask, j], mask ^ (1 << j)
            j = next(k for k in range(self.n) if prev >> k & 1 and dp[prev, k] + between[k, j] == value)
            mask = prev
            order.append(j)
        return tuple(reversed(order))
