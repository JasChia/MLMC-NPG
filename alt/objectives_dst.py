"""Scalarization for Deep Sea Treasure (Paper B eq. 46): f(R, t) = sqrt(R) + sqrt(50 - t).

J is the discounted return vector (M=2): J[0] = sum_t gamma^t r_R(t),
J[1] = sum_t gamma^t r_step(t), where r_step(t) = -1 each active step.

So J[1] is a "discounted negative-time" — at gamma=1, J[1] = -T (episode length).
For gamma close to 1 we approximate t (paper's "step count") as -J[1] and
compute f via the analytic limit.

We use a delta guard >> 0 so the gradient is bounded and MLMC's 2^Q
multiplier does not blow up when J[0] is briefly zero (policy hasn't found
any treasure yet) or when (50 + J[1]) approaches zero (bumping against the
T_max cap).

Note we cap (50 + J[1]) at >= 0 — if the policy uses a full 50 steps,
50 + J[1] = 50 - T = 0; the sqrt then equals sqrt(delta).
"""
from __future__ import annotations

import numpy as np


class DSTreasureObj:
    """f(J) = sqrt(J[0] + delta) + sqrt(max(0, 50 + J[1]) + delta).

    Maximizing this trades treasure value for fewer steps.
    """
    def __init__(self, delta: float = 1.0, T_max: int = 50):
        self.delta = float(delta)
        self.T_max = float(T_max)

    def value(self, J) -> float:
        J = np.asarray(J, dtype=np.float64)
        J0 = max(0.0, float(J[0]))
        time_term = max(0.0, self.T_max + float(J[1]))
        return float(np.sqrt(J0 + self.delta) + np.sqrt(time_term + self.delta))

    def grad(self, J) -> np.ndarray:
        J = np.asarray(J, dtype=np.float64)
        J0 = max(0.0, float(J[0]))
        time_term = max(0.0, self.T_max + float(J[1]))
        g = np.zeros(2, dtype=np.float64)
        g[0] = 0.5 / np.sqrt(J0 + self.delta)
        g[1] = 0.5 / np.sqrt(time_term + self.delta)  # +derivative w.r.t. J[1]
        return g
