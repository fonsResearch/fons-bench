"""Selective-prediction curves: risk vs coverage, AURC, and the normalised excess AURC.

Convention. A signal is oriented so that HIGHER means MORE confident. Accepting the top
fraction c of poses by that signal gives coverage c and risk = error rate among accepted.
`risk_coverage` returns the full curve evaluated at every achievable coverage (one point per
accepted pose), so AURC is an exact mean over those points rather than a grid approximation.

Definitions (also stated in results/kill_condition.md):
    AURC(signal) = mean over k of risk at coverage k/n, k = 1..n
    AURC(oracle) = the same with the poses sorted by the label (all correct first)
    AURC(random) = the constant error rate of the sample
    nE = (AURC_signal - AURC_oracle) / (AURC_random - AURC_oracle)
nE is 0 for perfect ordering and 1 for an uninformative signal; it can exceed 1 if the signal
is anti-correlated with correctness. It is undefined (NaN) when the sample is all-correct or
all-incorrect, because then every ordering has the same AURC.

Ties. Poses with equal signal values are ordered by a fixed random permutation seeded per call,
so ties neither help nor hurt systematically. `n_tie_groups` reports how much this matters.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RiskCoverage:
    coverage: np.ndarray      # k/n for k = 1..n
    risk: np.ndarray          # error rate among the top-k accepted
    aurc: float
    aurc_oracle: float
    aurc_random: float
    n: int
    n_errors: int
    n_tie_groups: int

    @property
    def excess(self) -> float:
        denom = self.aurc_random - self.aurc_oracle
        return float("nan") if denom <= 0 else (self.aurc - self.aurc_oracle) / denom

    def monotonicity_violation(self, min_coverage: float = 0.1) -> float:
        """Largest drop in risk as coverage grows, over coverages >= min_coverage.

        A perfectly behaved signal has risk non-decreasing in coverage, so this is 0. The value
        returned is max over c' > c of (risk(c) - risk(c')), i.e. how badly accepting *more*
        predictions lowers the observed error rate.
        """
        m = self.coverage >= min_coverage
        if m.sum() < 2:
            return float("nan")
        r = self.risk[m]
        return float(np.max(np.maximum.accumulate(r) - r))


def risk_coverage(signal: np.ndarray | pd.Series, correct: np.ndarray | pd.Series,
                  higher_is_better: bool = True, seed: int = 0) -> RiskCoverage:
    s = np.asarray(pd.Series(signal).astype(float))
    y = np.asarray(pd.Series(correct).astype(float))     # 1 = correct, 0 = error
    keep = ~np.isnan(s) & ~np.isnan(y)
    s, y = s[keep], y[keep]
    n = len(s)
    if n == 0:
        raise ValueError("no rows with both signal and label")
    if not higher_is_better:
        s = -s
    rng = np.random.default_rng(seed)
    order = np.lexsort((rng.permutation(n), -s))         # descending signal, ties broken randomly
    err = 1.0 - y[order]
    k = np.arange(1, n + 1)
    risk = np.cumsum(err) / k
    coverage = k / n
    n_err = int(err.sum())
    oracle_err = np.concatenate([np.zeros(n - n_err), np.ones(n_err)])
    aurc_oracle = float(np.mean(np.cumsum(oracle_err) / k))
    return RiskCoverage(coverage=coverage, risk=risk, aurc=float(np.mean(risk)),
                        aurc_oracle=aurc_oracle, aurc_random=float(n_err / n),
                        n=n, n_errors=n_err, n_tie_groups=int(n - len(np.unique(s))))


def bootstrap_aurc(signal, correct, n_boot: int = 1000, seed: int = 0, higher_is_better: bool = True) -> dict:
    """Percentile bootstrap CI for AURC and nE. Resamples poses with replacement."""
    s = np.asarray(pd.Series(signal).astype(float)); y = np.asarray(pd.Series(correct).astype(float))
    keep = ~np.isnan(s) & ~np.isnan(y); s, y = s[keep], y[keep]
    rng = np.random.default_rng(seed)
    a, e = [], []
    for i in range(n_boot):
        idx = rng.integers(0, len(s), len(s))
        rc = risk_coverage(s[idx], y[idx], higher_is_better, seed=i)
        a.append(rc.aurc); e.append(rc.excess)
    point = risk_coverage(s, y, higher_is_better, seed=seed)
    return {"aurc": point.aurc, "aurc_lo": float(np.percentile(a, 2.5)), "aurc_hi": float(np.percentile(a, 97.5)),
            "excess": point.excess, "excess_lo": float(np.nanpercentile(e, 2.5)), "excess_hi": float(np.nanpercentile(e, 97.5)),
            "n": point.n, "n_errors": point.n_errors}
