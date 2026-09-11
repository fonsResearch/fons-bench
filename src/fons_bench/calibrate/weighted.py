"""Weighted conformal risk control under covariate shift.

Setting: the calibration set is drawn from p_cal but the deployment population is p_test, and
the two differ in the shift coordinates x = (pocket similarity, ligand similarity, whether any
pre-cutoff neighbour exists). Under the standard covariate-shift assumption - the conditional
law of the label given x is unchanged, only the marginal of x moves - exchangeability is
restored by weighting each calibration point by the likelihood ratio w(x) = p_test(x)/p_cal(x).

We estimate w with a density-ratio classifier: label calibration points 0 and test points 1,
fit a probabilistic classifier d(x) = P(test | x) on the shift features only (never on the
signal, never on the label), and use w(x) = d(x) / (1 - d(x)) * (n_cal / n_test). The classifier
is a gradient-boosted tree with shallow depth; features are few and the mapping is smooth.

Thresholding. With weights, the acceptance rule uses a weighted version of the same walk as
`marginal`: descending by signal, the running weighted error rate must satisfy

    (sum of weights of incorrect points in top k + w_max) / (sum of weights in top k + w_max) <= alpha

where the w_max term plays the role of the "+1" test point in the unweighted case (the test
point's own weight is unknown, so the conservative choice is the largest calibration weight).
This is the standard weighted-quantile construction of Tibshirani et al. (2019), transposed to
selective risk.

Weights are clipped at `clip` times the mean before use. Unclipped likelihood ratios have
unbounded variance when the two populations barely overlap, and a single huge weight would let
one calibration point dictate the threshold. The clip fraction is reported so the reader can
see how much the estimate was tamed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_predict

from fons_bench.calibrate.base import Calibration

SHIFT_FEATURES = ["sucos_shape_pocket_qcov_cutoff", "morgan_tanimoto_cutoff", "n_targets_before_cutoff"]


def estimate_weights(cal: pd.DataFrame, test: pd.DataFrame, features: list[str] = SHIFT_FEATURES,
                     clip: float = 20.0, seed: int = 0) -> tuple[np.ndarray, dict]:
    """Likelihood ratios p_test/p_cal for the calibration rows, via a density-ratio classifier."""
    feats = [f for f in features if f in cal.columns and f in test.columns]
    X = pd.concat([cal[feats], test[feats]], ignore_index=True).astype(float)
    y = np.r_[np.zeros(len(cal)), np.ones(len(test))]
    clf = HistGradientBoostingClassifier(max_depth=3, max_iter=200, learning_rate=0.1, random_state=seed)
    # out-of-fold probabilities: an in-sample fit would overstate separability and inflate weights
    p = cross_val_predict(clf, X, y, cv=5, method="predict_proba")[:, 1]
    p_cal = np.clip(p[:len(cal)], 1e-6, 1 - 1e-6)
    w = (p_cal / (1 - p_cal)) * (len(cal) / max(len(test), 1))
    lim = clip * w.mean() if w.mean() > 0 else np.inf
    n_clipped = int((w > lim).sum())
    w = np.minimum(w, lim)
    auc_like = float(((p[len(cal):].mean()) - (p_cal.mean())))
    return w, {"features": feats, "n_clipped": n_clipped, "clip_limit": float(lim),
               "weight_mean": float(w.mean()), "weight_max": float(w.max()),
               "effective_sample_size": float(w.sum() ** 2 / np.sum(w ** 2)),
               "separability": auc_like}


def fit_threshold_weighted(signal, correct, weights, alpha: float) -> tuple[float, dict]:
    s = np.asarray(pd.Series(signal).astype(float))
    y = np.asarray(pd.Series(correct).astype(float))
    w = np.asarray(weights, dtype=float)
    keep = ~np.isnan(s) & ~np.isnan(y) & ~np.isnan(w)
    s, y, w = s[keep], y[keep], w[keep]
    n = len(s)
    if n == 0:
        return np.inf, {"n_cal": 0, "reason": "empty calibration set"}
    order = np.argsort(-s, kind="mergesort")
    s_, y_, w_ = s[order], y[order], w[order]
    w_max = w_.max()
    cum_w = np.cumsum(w_)
    cum_we = np.cumsum(w_ * (1.0 - y_))
    ok = (cum_we + w_max) / (cum_w + w_max) <= alpha
    if not ok.any():
        return np.inf, {"n_cal": n, "reason": "no threshold achieves alpha (weighted)"}
    k_star = int(np.max(np.arange(1, n + 1)[ok]))
    return float(s_[k_star - 1]), {"n_cal": n, "k_star": k_star, "w_max": float(w_max),
                                   "weighted_error_at_tau": float(cum_we[k_star - 1] / cum_w[k_star - 1])}


def calibrate_weighted(cal: pd.DataFrame, test: pd.DataFrame, alpha: float, signal: str,
                       label: str = "correct_num", clip: float = 20.0, seed: int = 0) -> Calibration:
    w, wdiag = estimate_weights(cal, test, clip=clip, seed=seed)
    tau, diag = fit_threshold_weighted(cal[signal], cal[label], w, alpha)
    return Calibration(method="weighted", alpha=alpha, tau=tau, n_cal=diag["n_cal"], notes={**diag, **wdiag})
