"""Generalized alpha-fairness scalarization.

Per-channel utility (Mo & Walrand 2000 isoelastic / power-mean fairness):

    U_alpha(x) = x^(1-alpha) / (1 - alpha)   for alpha != 1
    U_alpha(x) = log(x)                       for alpha == 1

Special cases:
    alpha = 0    : utilitarian (linear sum)
    alpha = 0.5  : 2*sqrt(.) fairness  [code's FruitTreeObj uses sqrt — half of canonical]
    alpha = 1    : log / proportional fairness
    alpha = 2    : -1/x fairness (TCP-style)
    alpha -> inf : max-min / Rawlsian (not differentiable; not used here)

Scalarization:
    f(J) = sum_m U_alpha( max(0, J_m) + delta )

`delta` is an additive guard so the gradient stays bounded near J_m = 0.

Concavity:
    alpha >= 0 -> U_alpha is concave; alpha = 0 is linear (still admissible).

Gradient w.r.t. J_m:  dU/dJ = x^(-alpha)  where x = max(0, J_m) + delta.
"""
from __future__ import annotations

import numpy as np


class AlphaFairObj:
    """f(J) = sum_m U_alpha(max(0, J_m) + delta).

    Parameters
    ----------
    alpha : float
        Fairness exponent. alpha=0 is utilitarian; alpha=0.5 is 2*sqrt;
        alpha=1 is log; alpha=2 is -1/x.
    delta : float
        Additive guard preventing unbounded gradient near J_m = 0.
    """

    def __init__(self, alpha: float, delta: float = 0.5):
        self.alpha = float(alpha)
        self.delta = float(delta)

    def value(self, J) -> float:
        J = np.asarray(J, dtype=np.float64)
        x = np.maximum(0.0, J) + self.delta
        if abs(self.alpha - 1.0) < 1e-12:
            return float(np.sum(np.log(x)))
        return float(np.sum(x ** (1.0 - self.alpha) / (1.0 - self.alpha)))

    def grad(self, J) -> np.ndarray:
        J = np.asarray(J, dtype=np.float64)
        x = np.maximum(0.0, J) + self.delta
        # dU_alpha/dx = x^(-alpha) for all alpha (including alpha=1 since
        # d/dx log(x) = 1/x = x^(-1)).
        return x ** (-self.alpha)
