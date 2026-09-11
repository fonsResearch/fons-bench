import numpy as np
import pytest

from fons_bench.risk_coverage import risk_coverage
from fons_bench.signals import assert_no_leakage


def test_perfect_signal_has_zero_excess():
    y = np.array([1, 1, 1, 0, 0])
    rc = risk_coverage(signal=np.array([5, 4, 3, 2, 1]), correct=y)
    assert rc.aurc == pytest.approx(rc.aurc_oracle)
    assert rc.excess == pytest.approx(0.0)


def test_inverted_signal_has_excess_above_one():
    y = np.array([1, 1, 1, 0, 0])
    rc = risk_coverage(signal=np.array([1, 2, 3, 4, 5]), correct=y)
    assert rc.excess > 1.0


def test_random_baseline_and_degenerate_labels():
    y = np.array([1, 0, 1, 0])
    rc = risk_coverage(signal=np.array([1, 1, 1, 1]), correct=y)   # all ties
    assert rc.aurc_random == pytest.approx(0.5)
    assert rc.n_tie_groups == 3
    allc = risk_coverage(signal=np.array([1.0, 2, 3]), correct=np.array([1, 1, 1]))
    assert np.isnan(allc.excess)                                    # undefined, not 0


def test_monotonicity_violation_detects_dip():
    # risk falls as coverage grows: the last accepted poses are all correct
    y = np.array([0, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    rc = risk_coverage(signal=np.arange(10)[::-1], correct=y)
    assert rc.monotonicity_violation(min_coverage=0.1) > 0.05


def test_leakage_guard_rejects_outcome_columns():
    assert_no_leakage(["ranking_score", "ranking_score_std"])
    for bad in ["rmsd", "lddt_pli", "correct", "pb_valid", "bb_rmsd"]:
        with pytest.raises(ValueError):
            assert_no_leakage([bad])


def test_dispersion_signals_are_oriented_low_is_confident():
    from fons_bench.signals import DISPERSION_SIGNALS, HIGHER_IS_BETTER, NATIVE_SIGNALS
    assert all(not HIGHER_IS_BETTER[c] for c in DISPERSION_SIGNALS)
    assert all(HIGHER_IS_BETTER[c] for c in NATIVE_SIGNALS)


def test_higher_is_better_flag_flips_the_curve():
    y = np.array([1, 1, 1, 0, 0])
    spread = np.array([1, 2, 3, 4, 5])          # low spread = correct
    assert risk_coverage(spread, y, higher_is_better=False).excess == pytest.approx(0.0)
    assert risk_coverage(spread, y, higher_is_better=True).excess > 1.0
