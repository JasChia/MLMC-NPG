"""Concave scalarizations f: R_+^M -> R applied to the return vector J.

Paper A (Bai et al. 2022), eq. (37) specifies:

    f(J_1, ..., J_M) = - sum_m  H / J_m

where J_m = sum_{t=1}^H gamma^{t-1} r_{m,t}. The H factor is cosmetic for
training (Adam normalizes gradient magnitude) but matters for aligning the
plotted objective values to Paper A's Figure 1.

We add a small delta > 0 in the denominator to guard against J_m -> 0 early
in training. This is a practical departure from the pure alpha-fairness
objective but keeps Lipschitz constants finite on the realized return set.
"""
from __future__ import annotations

import numpy as np


class AlphaFairness:
    """alpha = 2 case of alpha-fairness, scaled by H as in Paper A eq. (37).

    f(J)         = -H * sum_m 1 / (J_m + delta)
    grad_m f(J)  = +H        / (J_m + delta)^2
    """

    def __init__(self, alpha: float = 2.0, delta: float = 1e-3, H: int = 500):
        if not np.isclose(alpha, 2.0):
            raise NotImplementedError("Only alpha=2 implemented.")
        self.alpha = alpha
        self.delta = delta
        self.H = H

    def value(self, J) -> float:
        J = np.asarray(J, dtype=np.float64)
        return float(-self.H * np.sum(1.0 / (J + self.delta)))

    def grad(self, J) -> np.ndarray:
        J = np.asarray(J, dtype=np.float64)
        return self.H / (J + self.delta) ** 2
