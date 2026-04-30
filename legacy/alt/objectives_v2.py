"""Generalized alpha-fairness for any alpha >= 1.

f_alpha(J) = H * sum_m (J_m + delta)^{1 - alpha} / (1 - alpha)   [alpha > 1]
f_1(J)     = H * sum_m log(J_m + delta)                          [alpha = 1, limit]

For alpha = 2 this reduces to -H * sum_m 1/(J_m + delta), matching the
Paper A objective. Higher alpha is more concave in the small-J regime,
which increases the second-moment term in E[grad f(J_hat)] vs grad f(E[J_hat]) —
this is precisely the bias term MLMC is designed to remove.

alpha = 1 is the proportional-fairness limit, matching Paper B (Agarwal-
Aggarwal-Lan 2022) eq. 47 — the queueing scalarization. It is the unique
alpha at which Paper B's Assumption 6 (∂f is 2nd-order smooth) becomes
delta-dependent: |f''(J)| = H/(J+delta)^2, unbounded as delta -> 0.

grad_m f_alpha = H * (J_m + delta)^{-alpha}     (continuous through alpha=1)
"""
from __future__ import annotations

import numpy as np


class GeneralAlphaFairness:
    def __init__(self, alpha: float = 2.0, delta: float = 1e-3, H: int = 500):
        if alpha < 1.0:
            raise ValueError(f"alpha must be >= 1 (got {alpha})")
        self.alpha = float(alpha)
        self.delta = float(delta)
        self.H = int(H)

    def value(self, J) -> float:
        J = np.asarray(J, dtype=np.float64) + self.delta
        if self.alpha == 1.0:
            return float(self.H * np.sum(np.log(J)))
        return float(self.H * np.sum(np.power(J, 1.0 - self.alpha)) / (1.0 - self.alpha))

    def grad(self, J) -> np.ndarray:
        J = np.asarray(J, dtype=np.float64) + self.delta
        return self.H * np.power(J, -self.alpha)


class MinFairness:
    """f(J) = H * min_m J_m (max-min / egalitarian fairness).

    Lipschitz with constant H (in the L_inf norm) but NOT 2nd-order smooth: the
    sub-gradient is e_{argmin} (a discontinuous function of J). This is the
    canonical setting in which Paper B's Algorithm 1 (vanilla NPG, requires
    Assumption 6) loses its theoretical guarantee while Algorithm 2 (MLMC-NPG)
    retains it. Tie-breaking: split the unit gradient mass uniformly across
    all components within delta_tie of the minimum, so the gradient stays
    well-defined when several queues are simultaneously argmin.

    Use case: the policy must equalize J across queues, not merely maximize
    a smooth concave function — the harder Paper B Assumption-7 regime.
    """
    def __init__(self, H: int = 500, delta_tie: float = 1e-6):
        self.H = int(H)
        self.delta_tie = float(delta_tie)

    def value(self, J) -> float:
        J = np.asarray(J, dtype=np.float64)
        return float(self.H * J.min())

    def grad(self, J) -> np.ndarray:
        J = np.asarray(J, dtype=np.float64)
        m = J.min()
        ties = (J - m) <= self.delta_tie
        g = np.zeros_like(J)
        g[ties] = self.H / ties.sum()
        return g
