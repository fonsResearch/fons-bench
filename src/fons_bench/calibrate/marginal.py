"""Split conformal risk control, calibrated marginally on a random calibration split.

This is the naive baseline: one threshold for everything. Valid marginally over the pooled
population, and the point of the paper is that it is not valid conditionally on shift bin.

Construction. Sort calibration points by signal descending. For each candidate acceptance set
"top k", the empirical error rate is (# incorrect in top k) / k. We take the largest k whose
error rate satisfies the finite-sample bound

    (errors_k + 1) / (k + 1) <= alpha

The +1 in numerator and denominator is the standard conformal inflation for one unseen test
point: it makes the rule conservative at finite n rather than exactly-empirical, and it is what
turns "the training error rate happened to be <= alpha" into a guarantee that survives a fresh
draw. tau is then the signal value of the k-th point. If no k >= 1 qualifies, tau = +inf and
the rule accepts nothing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fons_bench.calibrate.base import Calibration


def fit_threshold(signal, correct, alpha: float) -> tuple[float, dict]:
    s = np.asarray(pd.Series(signal).astype(float))
    y = np.asarray(pd.Series(correct).astype(float))
    keep = ~np.isnan(s) & ~np.isnan(y)
    s, y = s[keep], y[keep]
    n = len(s)
    if n == 0:
        return np.inf, {"n_cal": 0, "reason": "empty calibration set"}
    order = np.argsort(-s, kind="mergesort")
    s_sorted, err_sorted = s[order], 1.0 - y[order]
    cum_err = np.cumsum(err_sorted)
    k = np.arange(1, n + 1)
    ok = (cum_err + 1.0) / (k + 1.0) <= alpha
    if not ok.any():
        return np.inf, {"n_cal": n, "reason": "no threshold achieves alpha", "base_error": float(err_sorted.mean())}
    k_star = int(np.max(k[ok]))
    tau = float(s_sorted[k_star - 1])
    # ties: any calibration point equal to tau is also accepted, so report the realised set size
    n_accept = int((s >= tau).sum())
    return tau, {"n_cal": n, "k_star": k_star, "n_accept_cal": n_accept,
                 "cal_error_at_tau": float((1.0 - y[s >= tau]).mean()),
                 "base_error": float(err_sorted.mean())}


def calibrate_marginal(cal: pd.DataFrame, alpha: float, signal: str, label: str = "correct_num") -> Calibration:
    tau, diag = fit_threshold(cal[signal], cal[label], alpha)
    return Calibration(method="marginal", alpha=alpha, tau=tau, n_cal=diag["n_cal"], notes=diag)
