"""Shared types for the three calibrations.

Framing (spec 6.1): this is conformal *risk control* on a selective classifier, not prediction
sets. Given a confidence signal s (higher = more confident) and a binary label
`correct in {0,1}`, choose a threshold tau such that among accepted predictions (s >= tau) the
probability of being incorrect is at most alpha. Report the abstention rate as the cost.

Nonconformity score. We use  r = -s , so that "large r" means "likely wrong". Selecting
s >= tau is the same as r <= -tau. Working in r keeps the quantile arithmetic in the usual
"upper quantile of the calibration scores" form.

Finite-sample guarantee. With n exchangeable calibration points, the split-conformal threshold
takes the ceil((n+1)(1-alpha))-th smallest score among the CORRECT calibration points. That
controls the error rate among accepted at level alpha in the usual conformal sense: it is a
guarantee about the *selection rule*, marginal over the draw of the calibration set, not a
per-test-point statement. When the required order statistic exceeds n, no finite threshold
achieves the level and the method must abstain from everything; we return tau = +inf and the
caller reports 100% abstention rather than silently accepting.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Calibration:
    """A fitted accept/reject rule."""
    method: str
    alpha: float
    tau: float | dict            # global threshold, or {group: threshold} for Mondrian
    n_cal: int | dict
    notes: dict = field(default_factory=dict)

    def accept(self, signal: np.ndarray, group: np.ndarray | None = None) -> np.ndarray:
        s = np.asarray(signal, dtype=float)
        if isinstance(self.tau, dict):
            if group is None:
                raise ValueError("Mondrian calibration needs group labels at test time")
            t = np.array([self.tau.get(g, np.inf) for g in np.asarray(group)], dtype=float)
        else:
            t = np.full(s.shape, float(self.tau))
        out = s >= t
        out[np.isnan(s)] = False            # no signal, no acceptance
        return out
