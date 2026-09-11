"""Tests for the three calibrations.

The important one is `test_marginal_is_marginally_valid_but_not_conditionally`: it builds a
synthetic two-bin population with the same structure as the real data (an easy bin and a hard
bin whose signal is less informative) and checks that a marginally-calibrated threshold meets
alpha overall while violating it inside the hard bin. That is the paper's claim in miniature,
verified against ground truth we control.
"""
import numpy as np
import pandas as pd
import pytest

from fons_bench.calibrate.base import Calibration
from fons_bench.calibrate.evaluate import evaluate
from fons_bench.calibrate.marginal import calibrate_marginal, fit_threshold
from fons_bench.calibrate.mondrian import calibrate_mondrian
from fons_bench.calibrate.weighted import calibrate_weighted, estimate_weights

ALPHA = 0.1


def _population(n, rng, hard_frac=0.5):
    """Two bins. In the easy bin the signal is strongly informative and a low-error acceptance
    region exists; in the hard bin the signal is weak and the base error rate is much higher.
    Both bins must contain a region where alpha is actually achievable, otherwise the correct
    behaviour of every method is to abstain everywhere and the tests measure nothing. The hard
    bin's admissible region is much smaller, which is what makes a single pooled threshold
    over-accept there."""
    hard = rng.uniform(size=n) < hard_frac
    s = rng.uniform(size=n)
    p_correct = np.where(hard, 0.05 + 0.93 * s ** 3, 0.35 + 0.64 * s ** 0.3)
    y = (rng.uniform(size=n) < p_correct).astype(float)
    return pd.DataFrame({"sig": s, "correct_num": y, "bin": np.where(hard, "hard", "easy"),
                         "sucos_shape_pocket_qcov_cutoff": np.where(hard, 20.0, 80.0) + rng.normal(0, 5, n),
                         "morgan_tanimoto_cutoff": rng.uniform(0, 100, n),
                         "n_targets_before_cutoff": rng.integers(0, 50, n)})


def test_threshold_respects_alpha_on_calibration_data():
    rng = np.random.default_rng(0)
    d = _population(4000, rng)
    tau, diag = fit_threshold(d.sig, d.correct_num, ALPHA)
    assert np.isfinite(tau)
    assert diag["cal_error_at_tau"] <= ALPHA + 1e-9


def test_no_achievable_threshold_returns_inf_and_accepts_nothing():
    d = pd.DataFrame({"sig": [0.9, 0.8, 0.7], "correct_num": [0.0, 0.0, 0.0]})
    cal = calibrate_marginal(d, ALPHA, "sig")
    assert cal.tau == np.inf
    assert not cal.accept(np.array([1.0, 0.99])).any()


def test_nan_signal_is_never_accepted():
    cal = Calibration(method="m", alpha=ALPHA, tau=0.5, n_cal=10)
    assert cal.accept(np.array([np.nan, 0.9])).tolist() == [False, True]


def test_marginal_is_marginally_valid_but_not_conditionally():
    """The paper's claim in miniature, over 40 independent draws.

    The conformal guarantee is marginal *in expectation over the calibration draw*, so on any
    single draw the pooled error straddles alpha - asserting it on one seed would be asserting
    noise. What is systematic is the contrast: pooled error sits at alpha on average, while the
    hard bin exceeds it on essentially every draw."""
    pooled, hard = [], []
    for seed in range(40):
        rng = np.random.default_rng(seed)
        cal_df, test_df = _population(6000, rng), _population(6000, rng)
        res = evaluate(calibrate_marginal(cal_df, ALPHA, "sig"), test_df, "sig", group="bin").set_index("group")
        pooled.append(res.loc["all", "realised_error"]); hard.append(res.loc["hard", "realised_error"])
    pooled, hard = np.array(pooled), np.array(hard)
    assert abs(pooled.mean() - ALPHA) < 0.02              # marginally valid on average
    assert (hard > ALPHA).mean() > 0.9                    # conditionally violated nearly always
    assert hard.mean() > pooled.mean() + 0.05             # and the violation is material


def test_mondrian_restores_conditional_validity_and_costs_abstention_where_it_binds():
    """Mondrian tightens the threshold in the bin that was violating alpha, and pays for it in
    abstention *there*. Note it can simultaneously relax the threshold in the easy bin, where a
    single pooled cut was needlessly strict, so total abstention may FALL. The cost of the
    guarantee is per-bin, not aggregate; the aggregate direction depends on the bin mix."""
    rng = np.random.default_rng(2)
    cal_df, test_df = _population(6000, rng), _population(6000, rng)
    marg = calibrate_marginal(cal_df, ALPHA, "sig")
    mond = calibrate_mondrian(cal_df, ALPHA, "sig", group="bin")
    rm = evaluate(marg, test_df, "sig", group="bin").set_index("group")
    ro = evaluate(mond, test_df, "sig", group="bin").set_index("group")
    assert rm.loc["hard", "realised_error"] > ALPHA + 0.02        # marginal was violating here
    assert ro.loc["hard", "realised_error"] < rm.loc["hard", "realised_error"]   # Mondrian repairs it
    assert ro.loc["hard", "abstention"] > rm.loc["hard", "abstention"]           # by abstaining more
    assert mond.tau["hard"] > marg.tau > mond.tau["easy"]         # stricter where hard, looser where easy


def test_mondrian_abstains_entirely_in_an_uncertifiable_group():
    d = pd.DataFrame({"sig": [0.9, 0.8, 0.7, 0.6], "correct_num": [0.0, 0.0, 0.0, 0.0], "bin": ["x"] * 4})
    cal = calibrate_mondrian(d, ALPHA, "sig", group="bin")
    assert cal.tau["x"] == np.inf


def test_weights_are_larger_where_test_population_is_denser():
    rng = np.random.default_rng(3)
    cal_df = _population(3000, rng, hard_frac=0.2)      # calibration mostly easy
    test_df = _population(3000, rng, hard_frac=0.8)     # deployment mostly hard
    w, diag = estimate_weights(cal_df, test_df)
    hard = (cal_df["bin"] == "hard").to_numpy()
    assert w[hard].mean() > w[~hard].mean()
    assert diag["effective_sample_size"] < len(cal_df)


def test_weighted_tightens_threshold_under_shift():
    rng = np.random.default_rng(4)
    cal_df = _population(4000, rng, hard_frac=0.2)
    test_df = _population(4000, rng, hard_frac=0.8)
    marg = calibrate_marginal(cal_df, ALPHA, "sig")
    wtd = calibrate_weighted(cal_df, test_df, ALPHA, "sig")
    assert wtd.tau >= marg.tau        # shifted deployment demands a stricter cut
