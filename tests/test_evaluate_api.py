"""Tests for the shipped calibration artifact and the `evaluate` entry point."""
import json

import numpy as np
import pandas as pd
import pytest

from fons_bench.evaluate_api import assign_bin, evaluate_predictions, load_calibration, summarise


def test_calibration_artifact_ships_and_is_wellformed():
    cal = load_calibration()
    assert cal["schema"] == "fons-bench-calibration-v1"
    assert cal["group_variable"] == "pocket_bin"
    assert len(cal["models"]) == 7
    for name, m in cal["models"].items():
        assert m["signal"] and m["n_calibration"] > 0
        for a, thr in m["alphas"].items():
            # a null threshold means "no achievable cut": legal, and must stay null not 0
            assert thr["marginal_tau"] is None or isinstance(thr["marginal_tau"], float)
            assert set(thr["mondrian_tau"]) <= {"0-30", "30-60", "60-90", "90-100"}


def test_uncertifiable_bins_are_present_in_the_artifact():
    """The Gate 3 finding must survive into the shipped thresholds, not be papered over."""
    cal = load_calibration()
    nulls = [(m, a, b) for m, e in cal["models"].items() for a, t in e["alphas"].items()
             for b, v in t["mondrian_tau"].items() if v is None]
    assert any(b == "0-30" and a == "0.1" for _, a, b in nulls)


def test_assign_bin_matches_edges():
    v = pd.Series([0, 29.9, 30, 89.9, 90, 100, None])
    out = assign_bin(v, [0, 30, 60, 90, 100])
    assert out.tolist()[:6] == ["0-30", "0-30", "30-60", "60-90", "90-100", "90-100"]
    assert out.iloc[6] is None


def _write(tmp_path, **cols):
    p = tmp_path / "preds.csv"
    pd.DataFrame(cols).to_csv(p, index=False)
    return p


def test_evaluate_without_pocket_similarity_only_does_marginal(tmp_path):
    p = _write(tmp_path, system_id=["a", "b", "c"], signal=[0.99, 0.5, 0.1])
    df = evaluate_predictions(p, "af3", 0.1)
    assert df["accept_conditional"].isna().all()
    assert df["accept_marginal"].dtype == bool


def test_evaluate_rejects_everything_in_an_uncertifiable_bin(tmp_path):
    p = _write(tmp_path, system_id=["a", "b"], signal=[0.999, 0.999], pocket_similarity=[5.0, 95.0])
    df = evaluate_predictions(p, "af3", 0.1).set_index("system_id")
    assert not df.loc["a", "accept_conditional"]      # 0-30 bin has no achievable threshold
    assert not df.loc["a", "bin_certifiable"]
    assert df.loc["b", "bin_certifiable"]


def test_unknown_model_and_alpha_fail_loudly(tmp_path):
    p = _write(tmp_path, system_id=["a"], signal=[0.9])
    with pytest.raises(SystemExit):
        evaluate_predictions(p, "no_such_model", 0.1)
    with pytest.raises(SystemExit):
        evaluate_predictions(p, "af3", 0.123)


def test_summarise_reports_abstention_and_realised_error(tmp_path):
    df = pd.DataFrame({"accept_marginal": [True, True, False], "accept_conditional": [True, False, False],
                       "correct": [1, 0, 1]})
    s = summarise(df, 0.1).set_index("rule")
    assert s.loc["marginal", "n_accepted"] == 2
    assert s.loc["marginal", "realised_error"] == pytest.approx(0.5)
    assert bool(s.loc["marginal", "exceeds_alpha"])
    assert s.loc["conditional", "abstention"] == pytest.approx(2 / 3)
